# Imports
import os
from re import L
import numpy as np;
import tensorflow as tf;
import pandas as pd;
import pickle;

from tensorflow import keras;
from keras.preprocessing.sequence import pad_sequences;
from tensorflow.keras.utils import to_categorical;
from keras.preprocessing.text import Tokenizer;
from tensorflow.python.keras.activations import relu, sigmoid, softmax;
from tensorflow.python.ops.gen_nn_ops import leaky_relu;
from keras.layers import Bidirectional, LSTM;

# Getting data
data = pd.read_csv("shortjokes.csv")["Joke"];

# Creating Tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<>");
tokenizer.fit_on_texts(data);

# Getting training data 
train_x = tokenizer.texts_to_sequences(pd.read_csv("shortjokes.csv")["Joke"][0:250]);# tokenizer.texts_to_sequences(pd.read_csv("movie_reviews_full.csv")["reviews"][0:50]);
formatted_train_x = [];
train_y = [];
maxlen = 0;

for i in train_x:
    maxlen = max(len(i), maxlen);
    
    if (len(i) > 1):
        for v in range(1, len(i)):
            formatted_train_x.append(i[:v]);
            train_y.append(i[v]);

formatted_train_x = pad_sequences(formatted_train_x, maxlen=maxlen);
train_y = to_categorical(train_y, num_classes=len(tokenizer.word_index));

# Main 
model = keras.Sequential([
    keras.layers.Embedding(len(tokenizer.word_index), 240, input_length=maxlen),
    keras.layers.Bidirectional(keras.layers.LSTM(150)),
    keras.layers.Dense(len(tokenizer.word_index), activation=softmax),
]);     

model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=["accuracy"]);
model.fit(formatted_train_x, train_y, epochs=100);
model.save("model.h5");

with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL);

print(maxlen)