from django.db import models

# Create your models here.

class texto(models.Model):
  frase = models.CharField(max_length=30)

  def __str__(self):
    return self.frase