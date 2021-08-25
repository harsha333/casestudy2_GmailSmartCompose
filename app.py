from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import numpy as np
import re
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

app = Flask(__name__)

######################################################################
final_model = load_model("final_model.h5", compile=False)
test_model = load_model("test_model.h5", compile=False)

max_len_size_vocab = 10000

with open('word_dict-final.json') as f:
    dict_words = json.load(f)
    toknizer = Tokenizer(filters='', num_words=max_len_size_vocab)
    toknizer.word_index = dict_words

max_len_in = 21
max_len_out = 20
def test_tokenize_data(data):
    data = data.lower()
    data = '<start> ' + data + ' <end>'
    data_tf = toknizer.texts_to_sequences([data])
    data_tf = pad_sequences(data_tf, maxlen=max_len_in, padding="post")
    return data_tf

integer2words = map(reversed, toknizer.word_index.items())
integer2words = dict(integer2words)
def get_testoutput(query_point):
    st = final_model.predict(query_point)
    sequence_output = np.zeros((1, 1))
    sequence_output[0, 0] = toknizer.word_index['<start>']
    present = "<start>"
    sen_final = ''
    k = 0
    while present != "<end>" and k < (max_len_out - 1):
        tkns_final, st_h = test_model.predict([sequence_output, st])

        tkn_present = np.argmax(tkns_final[0, 0])

        if (tkn_present == 0):
          break;

        present = integer2words[tkn_present]

        sen_final += ' ' + present
        sequence_output[0, 0] = tkn_present
        st = st_h
        k += 1

    return sen_final

def getWords(w):
    res=''
    if w!=0:
      res = integer2words[w]
    else:
      res = ''
    return res

def tkns2seq(inputs):
  sentnces = list(map(getWords, inputs))
  result = ' '.join(sentnces)
  return result
###########################################################################
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_seq = request.form['inp_seq']
    print(input_seq)
    