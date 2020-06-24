from django.db import models

# Create your models here.
class Reply(models.Model):
    user = models.CharField(max_length=20)
    time = models.CharField(max_length=20)
    question = models.CharField(max_length=50)
    answer = models.CharField(max_length=50)
