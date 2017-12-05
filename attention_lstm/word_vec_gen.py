import keras.backend as K
import numpy as np
import collections
from nltk.corpus import sentiwordnet as swn
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
input_dim = 25       # should be changed to 300 or whatever is the word embedding size
time_steps = 20

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

time_steps=20
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
file_cnt=1;
null_cnt=0;
for i in range(len(input_data)):
    temp=[]
    sentiments=[]
    for j in range(len(input_data[i])):
        if(input_data[i][j]!=0):
            current=Glove.get(reverse_dictionary.get(input_data[i][j],'UNK'),dummy_vector);
            if(not(current==dummy_vector)):
                sentiment.append()
                temp.append(current);
    with open(GLOVE_DIR+"word_embeddings_50d/"+str(file_cnt)+".txt","w+", encoding="utf-8") as out:
        for i in range(len(temp)):
            for j in temp[i]:
                out.write(str(j));
                out.write(' ');
            out.write("\n");
    file_cnt+=1
    null_cnt+=max_seq_len-len(temp);
    for j in range(max_seq_len-len(temp)):
        temp.append(dummy_vector);
    temp=temp[:][0:max_seq_len];
    X.append(temp);
print('Total number of null vectors is: '+str(null_cnt));
for i in range(len(X)):
    assert len(X[i])==max_seq_len
Y=[]
with open(labels_file,'r',encoding="utf-8") as inp, open("labels_50d.txt","w",encoding="utf-8") as out:
    for line in inp:
        Y.append(str(line.split()[0]))
        out.write(str(line.split()[0]));
        out.write("\n");         
        
print('checking input and output sizes')
assert len(Y)==len(X)
#x=np.asarray(X);
#y=np.asarray(Y);
#print('Shapes of inputs and outputs: ');
#print(x.shape);
#print(y.shape);
#return x, y
