# Imports
import os;
import numpy as np;
import tensorflow as tf;
import pandas as pd;
import numpy as np;
import pickle;

from tensorflow import keras;

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer;

# Variables
maxlen = 40;
# Creating Tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle);

# Main
model = keras.models.load_model("model.h5");

seed = "Three guys walked into a bar".split();

for x in range(20):
    raw = model.predict(pad_sequences(tokenizer.texts_to_sequences([seed]), maxlen=maxlen))
    predict = np.argmax(raw[0]);

    for i in tokenizer.word_index:
        if (tokenizer.word_index[i] == predict):
            predict = i;
            break;

    seed.append(predict);

print(seed)