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
import socket
from _thread import *


## bert 모델 불러오기, vocab 불러오기, tokenizer 불러오기..
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

## 필요한 파라미터
vocab_size = len(vocab.idx_to_token)
max_len = 64

## 모델 클래스
class BERTseq2seq(nn.Module):
    def __init__(
        self, bert, hidden_size=768, vocab_size=8002, dr_rate=None, params=None
    ):
        super(BERTseq2seq, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

        self.linear = nn.Linear(hidden_size, vocab_size)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        seq, _ = self.bert(
            input_ids=token_ids,
            token_type_ids=segment_ids.long(),
            attention_mask=attention_mask.float().to(token_ids.device),
        )
        # if self.dr_rate:
        #   out = self.dropout(seq)
        seq = self.linear(seq)

        return seq


## 모델 초기화
model = BERTseq2seq(bertmodel, vocab_size=vocab_size, dr_rate=0.5).to(device)

## 모델 불러오기
save_path = "./model/sample_model.tar"
checkpoint = torch.load(save_path)
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
    seq = model(token.long(), length, segment.long())
    _, topi = seq.topk(1)
    answer = ""
    for x in topi[0]:
        answer += vocab.idx_to_token[x.item()]
        if x.item() in [1, 3]:
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
