# from xgboost import XGBClassifier
# from xgboost import XGBRegressor
# from sklearn.cross_validation import train_test_split
# from sklearn.metrics import accuracy_score
# import re
# import numpy as np
import os
# from nltk.tokenize import TweetTokenizer

from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import Booster
from xgboost import train
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from nltk.tokenize import TweetTokenizer
import re
from nltk.corpus import sentiwordnet as swn
import numpy as np

# import keyword_extraction_w_parser

from sklearn.feature_extraction.text import TfidfVectorizer

def parse_dataset(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with open(fp, 'rt') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"):  # discard first line if it contains metadata
                line = line.rstrip()  # remove trailing whitespace
                split_line = line.split("\t")
                label = int(split_line[1])
                tweet = split_line[2]
                y.append(label)
                corpus.append(tweet)

    return corpus, y

DATASET_FP = "./SemEval2018-T4-train-taskA.txt"
TASK = "A"  # Define, A or B
FNAME = './predictions-task' + TASK + '.txt'
PREDICTIONSFILE = open(FNAME, "w")
corpus, inp_y = parse_dataset(DATASET_FP)

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize

MAX_LEN_2 = 6
one_hot_y = []
inp_X = []

def get_sent_max_pos_neg(corpus):
    for curr_sentence in corpus:
        curr_sentence = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', curr_sentence)
        curr_sentence = re.sub(r'@[a-zA-Z0-9_]+', ' ', curr_sentence)
        sentence_tokenized = tokenizer(curr_sentence)
        sent_list = []
        for i in range(len(sentence_tokenized)-2):
            curr_word = sentence_tokenized[i]
            next_word = sentence_tokenized[i+1]
            next_next = sentence_tokenized[i+2]
            if curr_word.startswith('#'):
                curr_word = curr_word[1:]
            if next_word.startswith('#'):
                next_word = next_word[1:]
            if next_next.startswith('#'):
                next_next = next_next[1:]
            curr_senti_synsets = swn.senti_synsets(curr_word)
            next_senti_synsets = swn.senti_synsets(next_word)
            next_next_senti_synsets = swn.senti_synsets(next_next)
            if len(curr_senti_synsets) > 0 and len(next_senti_synsets) > 0 and len(next_next_senti_synsets) > 0:
                curr_pos = curr_senti_synsets[0].pos_score()
                curr_neg = curr_senti_synsets[0].neg_score()
                next_pos = next_senti_synsets[0].pos_score()
                next_neg = next_senti_synsets[0].neg_score()
                next_next_pos = next_next_senti_synsets[0].pos_score()
                next_next_neg = next_next_senti_synsets[0].neg_score()
                max_pos = max(curr_pos, next_pos, next_next_pos)
                max_neg = max(curr_neg, next_neg, next_next_neg)
                curr_sent = max_pos - max_neg
                if curr_sent != 0:
                    sent_list.append(curr_sent)
        if len(sent_list) > MAX_LEN_2:
            sent_list = sent_list[:MAX_LEN_2]
        for i in range(MAX_LEN_2 - len(sent_list)):
            sent_list.append(0.0)
        inp_X.append(sent_list)
    return inp_X

inp_X = get_sent_max_pos_neg(corpus)
inp_X = np.array(inp_X)
##############################################


#preprocess data files that contain word_embedding features.
#May need to change path based on where data is
data_y_fp = "./labels.txt"
directory = "word_embeddings"
#350 best for 25d, 550 best for 50d
MAX_LENGTH = 350
data_x = []
data_y = []

#go through labels file:
with open(data_y_fp) as f:
	for label in f:
		label = int(label)
		data_y.append(label)
data_y = np.array(data_y)

print "test"
#go through the feature files. Each file has 25 features
files = os.listdir(directory)
os.chdir(directory)
for filename in files:
    example = []
    name = filename.split(".")
    file_num = int(name[0])
    label = data_y[file_num-1]
    sent_scores = inp_X[file_num-1]
    with open(filename) as f:
    	for line in f:
    		tmp = line.split(" ")
    		tmp = tmp[:len(tmp)-1]
    		tmp = map(float, tmp)
    		example = example + tmp
    example = np.array(example)
    if len(example)>MAX_LENGTH:
    	example = example[:MAX_LENGTH]
    if len(example)<MAX_LENGTH:
    	example = np.pad(example, (0,MAX_LENGTH-len(example)), 'constant')
    example = np.append(example, sent_scores)
    example = np.append(example, label)
    data_x.append(example)
data_x = np.array(data_x)
print "data_x dims"
print data_x.shape
X = data_x[:,0:MAX_LENGTH+MAX_LEN_2]
Y = data_x[:,MAX_LENGTH+MAX_LEN_2]

seed = 4
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=test_size)
# fit model no training data
# model = XGBClassifier(max_depth = 3)
model = XGBRegressor(max_depth=3) #gave 56.51%
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

