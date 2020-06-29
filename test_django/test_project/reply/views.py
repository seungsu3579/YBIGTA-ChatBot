from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
import socket
import json
from django.views.decorators.csrf import csrf_exempt

def transform_reply(answer):

    answer = answer.strip("[CLS]").strip("[PAD]").strip("[SEP]").strip("▁")
    answer = answer.replace("▁", " ")

    return answer

def block_func(question):
    answer = ""

    if question.find("대면") != -1:
        answer = "면접과 방학세션은 비대면으로 진행됩니다"
    
    if question.find("경쟁률") != -1:
        answer = "경쟁률은 알려드릴 수 없는 점 양해 부탁드립니다"

    return answer

@csrf_exempt
def reply(request):

    if request.method == "POST":
        body = ((request.body).decode('utf-8'))
        form = json.loads(body)
        data = form['userRequest']['utterance']

        if block_func(data) == "":
        
            HOST = "127.0.0.1"
            PORT = 9999

            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((HOST, PORT))
        
            client_socket.sendall(data.encode())
            data = client_socket.recv(1024)
            data = data.decode()
        else:
            data = block_func(data)
        

    response = {
            "contents": [
                {
                    "type": "text",
                    "text": transform_reply(data)
                }
            ]
        }

    jsonObj = json.dumps(response, ensure_ascii=False)

    return HttpResponse(jsonObj, content_type='application/json; charset = utf-8')
