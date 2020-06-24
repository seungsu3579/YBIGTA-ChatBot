from rest_framework import viewsets
from .serializers import ReplySerializer
from models import Reply

# Create your views here.
class ReplyViewSet(viewsets.ModelViewSet):
    queryset = Reply.objects.create(name="seungsu")
    serializer_class = ReplySerializer
