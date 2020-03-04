from rest_framework import serializers
from . models import texto

class textoSerializers(serializers.ModelSerializer):
  class Meta:
    model = texto
    fields=('__all__')