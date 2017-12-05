import keras.backend as K
import numpy as np
import collections
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#import collections
#import math
#import os
#import random
#from tempfile import gettempdir
#import zipfile
#from six.moves import urllib
#from six.moves import xrange 
#import tensorflow as tf

GLOVE_DIR='C:/Users/Dell/Documents/MS/Codes/irony_detection/RNNs/keras-attention-mechanism-master/'
inputs_file='C:/Users/Dell/Documents/MS/Codes/irony_detection/RNNs/keras-attention-mechanism-master/SE18-taskA_inputs.txt'
labels_file='C:/Users/Dell/Documents/MS/Codes/irony_detection/RNNs/keras-attention-mechanism-master/SE18-taskA_labels.txt'
embeddings_index = {}
vocab_size=15970

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  print('Dictionary length:');
  print(str(len(dictionary))+' '+str(len(reversed_dictionary)));
  return count, dictionary, reversed_dictionary


# loading Glove model
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r',encoding="utf-8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print("Done."+str(len(model))+"words loaded!")
    return model

def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def get_data(n, input_dim, attention_column=1):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column] = y[:, 0]
    return x, y


def get_data_recurrent(n, time_steps, input_dim, attention_column=10):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    max_seq_len=time_steps
    words=[]
    with open(inputs_file,'r',encoding="utf-8") as inp:
        for line in inp:
            words.extend(line.split());
    print('Total number of words: '+str(len(words)))
    # data - list of codes (integers from 0 to vocabulary_size-1).
    #   This is the original text but words are replaced by their codes
    # count - map of words(strings) to count of occurrences
    # dictionary - map of words(strings) to their codes(integers)
    # reverse_dictionary - maps codes(integers) to words(strings)
    count, dictionary, reverse_dictionary = build_dataset(words,vocab_size)
    input_data=[]
    with open(inputs_file,'r',encoding="utf-8") as inp:
        for line in inp:
            input_data.append([dictionary.get(word, 0) for word in line.split()]);
    X=[]
    Glove=loadGloveModel(GLOVE_DIR+'glove.twitter.27B.50d.txt');
    dummy_vector=[0.0 for i in range(input_dim)];
    print('Shape of dummy vector: '+str(len(dummy_vector)));
    null_cnt=0;
    for i in range(len(input_data)):
        temp=[]
        for j in range(len(input_data[i])):
            if(input_data[i][j]!=0):
                current=Glove.get(reverse_dictionary.get(input_data[i][j],'UNK'),dummy_vector);
                if(not(current==dummy_vector)):
                    temp.append(current);
        null_cnt+=max_seq_len-len(temp);
        for j in range(max_seq_len-len(temp)):
            temp.append(dummy_vector);
        temp=temp[:][0:max_seq_len];
        X.append(temp);
    print('Total number of null vectors is: '+str(null_cnt));
    for i in range(len(X)):
        assert len(X[i])==max_seq_len
    Y=[]
    with open(labels_file,'r',encoding="utf-8") as inp:
        for line in inp:
            Y.append(str(line.split()[0]))
    print('checking input and output sizes')
    assert len(Y)==len(X)
    x=np.asarray(X);
    y=np.asarray(Y);
    print('Shapes of inputs and outputs: ');
    print(x.shape);
    print(y.shape);
    return x, y
