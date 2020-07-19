# -*- coding: utf-8 -*-
## 필요 패키지 로드
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
from kobert.pytorch_kobert import get_pytorch_kobert_model
import pandas as pd
import numpy as np
from kobert.utils import get_tokenizer
from transformers import AdamW
from transformers.optimization import WarmupLinearSchedule
import random
import socket
from _thread import *

device = "cuda" if torch.cuda.is_available() else "cpu"

## bert 모델 불러오기, vocab 불러오기, tokenizer 불러오기..
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

## 하이퍼 파라미터 설정
max_len = 100
batch_size = 32
warmup_ratio = 0.1
num_epochs = 160
num_workers = 0
max_grad_norm = 1
learning_rate = 5e-5
print_every = 150
save_every = 20
embed_dim = 100
vocab_size = len(vocab.idx_to_token)

## 모델 클래스
class Encoder(nn.Module):
    def __init__(self, bert):
        super(Encoder, self).__init__()
        self.bert = bert

    def gen_attention_mask(self, token_ids, valid_length):
        ## masked attenion, 패딩에 패널티 부여, 학습 x
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        output, hidden = self.bert(
            input_ids=token_ids,
            token_type_ids=segment_ids.long(),
            attention_mask=attention_mask.float().to(token_ids.device),
        )

        return output, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_in = enc_hid_dim + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(
            self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2))
        )

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(
        self,
        enc_hid_dim,
        hidden_dim,
        emb_dim,
        vocab_size,
        attention,
        max_length,
        dropout_rate=0.1,
    ):
        super(Decoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim

        self.attention = attention

        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.gru = nn.GRU(
            self.enc_hid_dim + self.emb_dim, self.hidden_dim, batch_first=True
        )
        self.out = nn.Linear(self.attention.attn_in, self.vocab_size)

    def _weighted_encoder_rep(self, decoder_hidden, encoder_outputs):
        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        return weighted_encoder_rep

    def forward(self, input, decoder_hidden, encoder_outputs):

        input = input.unsqueeze(1).long()
        embedded = self.embedding(input)
        weighted_encoder_rep = self._weighted_encoder_rep(
            decoder_hidden, encoder_outputs
        )

        input = torch.cat((embedded, weighted_encoder_rep), dim=2)

        output, decoder_hidden = self.gru(input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(1)

        output = self.out(torch.cat((output, weighted_encoder_rep), dim=1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.vocab_size = self.decoder.vocab_size

    def forward(self, src, valid_length, segment_ids, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        max_len = trg.shape[1]

        encoder_outputs, hidden = self.encoder(src, valid_length, segment_ids)

        outputs = torch.zeros(max_len, batch_size, self.vocab_size).to(self.device)

        output = trg[:, 0].long()

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = trg[:, t] if teacher_force else top1

        return outputs.permute(1, 2, 0)


## 모델 초기화
encoder = Encoder(bertmodel)
attention = Attention(768, 768, 100)
decoder = Decoder(768, 768, embed_dim, vocab_size, attention, max_len)
model = Seq2Seq(encoder, decoder, device).to(device)


## 모델 불러오기
save_path = "./model/checkpoint_2_sample.tar"
checkpoint = torch.load(save_path, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])


## 입력된 텍스트에 대한 답변 리턴
def translate(s):

    transform = nlp.data.BERTSentenceTransform(
        tok, max_seq_length=max_len, pad=True, pair=False
    )
    token, length, segment = transform([s])
    token = torch.from_numpy(token).to(device)
    length = torch.from_numpy(length).to(device)
    segment = torch.from_numpy(segment).to(device)
    model.eval()
    token = token.view(1, -1)
    length = length.view(1, -1)
    segment = segment.view(1, -1)
    label = torch.zeros(1, max_len).to(device)
    label.fill_(2)
    seq = model(token.long(), length, segment.long(), label, teacher_forcing_ratio=0)
    seq = seq.permute(0, 2, 1)
    _, topi = seq.topk(1)
    answer = ""
    for x in topi[0]:
        answer += vocab.idx_to_token[x.item()]
        if x.item() in [3]:
            break

    return answer


# 접속한 클라이언트마다 새로운 쓰레드가 생성되어 통신을 하게 됩니다.
def threaded(client_socket, addr):

    print("Connected by :", addr[0], ":", addr[1])

    # 클라이언트가 접속을 끊을 때 까지 반복합니다.
    while True:

        try:

            # 데이터가 수신되면 클라이언트에 다시 전송합니다.(에코)
            data = client_socket.recv(1024)

            if not data:
                print("Disconnected by " + addr[0], ":", addr[1])
                break

            print("Received from " + addr[0], ":", addr[1], data.decode())

            #########################################################
            # 응답 함수 콜
            recieve_message = data.decode()
            answer = translate(recieve_message)
            #########################################################

            # 에코
            client_socket.send(answer.encode())

        except ConnectionResetError as e:

            print("Disconnected by " + addr[0], ":", addr[1])
            break

    client_socket.close()


HOST = "127.0.0.1"
PORT = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()

print("server start")


# 클라이언트가 접속하면 accept 함수에서 새로운 소켓을 리턴합니다.

# 새로운 쓰레드에서 해당 소켓을 사용하여 통신을 하게 됩니다.
while True:

    print("wait")

    client_socket, addr = server_socket.accept()
    start_new_thread(threaded, (client_socket, addr))

server_socket.close()
