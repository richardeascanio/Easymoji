from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import api_view
from django.core import serializers
from rest_framework.response import Response 
from rest_framework import status 
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from . models import texto
from . serializers import textoSerializers
import pickle
from sklearn.externals import joblib
import json
import numpy as np
from sklearn import preprocessing
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import keras
from keras.backend import set_session
from . emo_utils import *
import spacy
from django.core.cache import cache

# Create your views here.

class TextoView(viewsets.ModelViewSet):
  queryset = texto.objects.all()
  serializer_class = textoSerializers

def sentences_to_indices2(X, max_len):
  
  m = X.shape[0]
  print(m)

  X_indices = np.zeros((m, max_len), dtype=int)

  # loop sobre el training data
  for i in range(m):
    frase = X[i]
    frase = str(frase)
    tokens = nlp(frase)

    # Inicializamos j en 0
    j = 0

    # Loop sobre las palabras de la oracion
    for token in tokens:
      if 'jaja' in token.text or 'JAJA' in token.text:
        X_indices[i, j] = 2444
      elif '?' in token.text or '¿' in token.text:
        X_indices[i, j] = 1748
      elif '!' in token.text or '¡' in token.text:
        X_indices[i, j] = 4032
      else:
        if token.has_vector:
          X_indices[i, j] = nlp.vocab.vectors.key2row[token.norm] # pasar de string a numero
        else:
          X_indices[i, j] = 0
      j += 1

  return X_indices

def getModel():
  loaded_model = joblib.load("C:\\Users\\rasca\\OneDrive\\Escritorio\\emojiModelFlattenBidirectional.pkl")
  return loaded_model

graph = tf.get_default_graph()
sess = keras.backend.get_session()
init = tf.global_variables_initializer()
sess.run(init)
model = getModel()
print('Model loaded')

def rankText(text):
  x_test = np.array([text])
  print('x_test', x_test)
  print('shape del x_test', x_test.shape)  # shape (1,)
  global sess
  global graph
  with graph.as_default():
    set_session(sess)
    X_test_indices = sentences_to_indices2(x_test, 47)
    print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))
    # pedimos al modelo los mejores tres emojis junto con su porcentaje
    print(label_to_emoji(np.argsort(model.predict(X_test_indices)).item(21)) + ' con un porcentaje de:',model.predict(X_test_indices).item(np.argsort(model.predict(X_test_indices)).item(21)))
    print(label_to_emoji(np.argsort(model.predict(X_test_indices)).item(20)) + ' con un porcentaje de:',model.predict(X_test_indices).item(np.argsort(model.predict(X_test_indices)).item(20)))
    print(label_to_emoji(np.argsort(model.predict(X_test_indices)).item(19)) + ' con un porcentaje de:',model.predict(X_test_indices).item(np.argsort(model.predict(X_test_indices)).item(19)))
    # creamos un array donde guardar los tres mejores emojis junto con su respectivo porcentaje
    emojis = []
    emojis.append(label_to_emoji(np.argsort(model.predict(X_test_indices)).item(21)))
    emojis.append(model.predict(X_test_indices).item(np.argsort(model.predict(X_test_indices)).item(21)))
    emojis.append(label_to_emoji(np.argsort(model.predict(X_test_indices)).item(20)))
    emojis.append(model.predict(X_test_indices).item(np.argsort(model.predict(X_test_indices)).item(20)))
    emojis.append(label_to_emoji(np.argsort(model.predict(X_test_indices)).item(19)))
    emojis.append(model.predict(X_test_indices).item(np.argsort(model.predict(X_test_indices)).item(19)))
    print(emojis)

  return emojis

# Metodo que se va a ejecutar al hacer una peticion POST al servidor /status
@api_view(["POST"])
def textoreject(request):
  try:
    print('modelo', model)
    print(request)
    mydata = JSONParser().parse(request)
    # my data es el dict (json) que contiene el texto que queremos analizar
    print('my data',mydata)
    print('tipo my data', type(mydata))
    # tomamos el valor del atributo 'texto' del dict
    texto = mydata.get("texto")
    print('texto', texto)
    print(type(texto))
    # metemos ese string en un numpy array para probar el modelo
    # x_test = np.array([texto])
    # print('x_test', x_test)
    # print('shape del x_test', x_test.shape)  # shape (1,)
    # # llamamos al modelo para convertir el string en un array de indices de longitud maxLen (39) en este caso
    # X_test_indices = sentences_to_indices2(x_test, 52)
    # # mandamos el array al metodo de prediccion del modelo
    # # nos devuelve el id del emoji, lo pasamos por label_to_emoji para obtener la carita correspondiente
    # print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))
    # # pedimos al modelo los mejores tres emojis junto con su porcentaje
    # print(label_to_emoji(np.argsort(model.predict(X_test_indices)).item(21)) + ' con un porcentaje de:',model.predict(X_test_indices).item(np.argsort(model.predict(X_test_indices)).item(21)))
    # print(label_to_emoji(np.argsort(model.predict(X_test_indices)).item(20)) + ' con un porcentaje de:',model.predict(X_test_indices).item(np.argsort(model.predict(X_test_indices)).item(20)))
    # print(label_to_emoji(np.argsort(model.predict(X_test_indices)).item(19)) + ' con un porcentaje de:',model.predict(X_test_indices).item(np.argsort(model.predict(X_test_indices)).item(19)))
    # # creamos un array donde guardar los tres mejores emojis junto con su respectivo porcentaje
    # emojis = []
    # emojis.append(label_to_emoji(np.argsort(model.predict(X_test_indices)).item(21)))
    # emojis.append(model.predict(X_test_indices).item(np.argsort(model.predict(X_test_indices)).item(21)))
    # emojis.append(label_to_emoji(np.argsort(model.predict(X_test_indices)).item(20)))
    # emojis.append(model.predict(X_test_indices).item(np.argsort(model.predict(X_test_indices)).item(20)))
    # emojis.append(label_to_emoji(np.argsort(model.predict(X_test_indices)).item(19)))
    # emojis.append(model.predict(X_test_indices).item(np.argsort(model.predict(X_test_indices)).item(19)))
    # print(emojis)

    emojis = rankText(texto)


    # mandamos el array de emojis como respuesta al front end.
    return JsonResponse(emojis, safe=False)
  except ValueError as e:
    return Response(e.args[0], status.HTTP_400_BAD_REQUEST)
