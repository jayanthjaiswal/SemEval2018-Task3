# First XGBoost model for Pima Indians dataset
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

import keyword_extraction_w_parser

from sklearn.feature_extraction.text import TfidfVectorizer

# load data
#dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
#X = dataset[:,0:8]
#Y = dataset[:,8]

#preprocess tweet data:
# Dataset: SemEval2018-T4-train-taskA.txt or SemEval2018-T4-train-taskB.txt


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

MAX_LEN = 10
one_hot_y = []
inp_X = []

#features (change - look at sentiments between neighbors)
def get_sent_og(corpus):
	for curr_sentence in corpus:
		curr_sentence = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', curr_sentence)
		curr_sentence = re.sub(r'@[a-zA-Z0-9_]+', ' ', curr_sentence)
		sentence_tokenized = tokenizer(curr_sentence)
		sent_list = []
		for curr_word in sentence_tokenized:
			if curr_word.startswith('#'):
				curr_word = curr_word[1:]
			curr_senti_synsets = swn.senti_synsets(curr_word)
			if len(curr_senti_synsets) > 0:
				c_pos = curr_senti_synsets[0].pos_score()
				c_neg = curr_senti_synsets[0].neg_score()
				c_obj = curr_senti_synsets[0].obj_score()
				# c_subj = max(1.0 - c_obj, 0.01)
				curr_sent = (c_pos - c_neg)
				if curr_sent != 0:
					sent_list.append(curr_sent)
		if len(sent_list) > MAX_LEN:
			sent_list = sent_list[:MAX_LEN]
		for i in range(MAX_LEN - len(sent_list)):
			sent_list.append(0.0)
		inp_X.append(sent_list)
	return inp_X

#comparing positive scores between pairs of words
def get_sent_pos_comp(corpus):
	for curr_sentence in corpus:
		curr_sentence = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', curr_sentence)
		curr_sentence = re.sub(r'@[a-zA-Z0-9_]+', ' ', curr_sentence)
		sentence_tokenized = tokenizer(curr_sentence)
		sent_list = []
		for i in range(len(sentence_tokenized)-1):
			curr_word = sentence_tokenized[i]
			next_word = sentence_tokenized[i+1]
			if curr_word.startswith('#'):
				curr_word = curr_word[1:]
			if next_word.startswith('#'):
				next_word = next_word[1:]

			curr_senti_synsets = swn.senti_synsets(curr_word)
			next_senti_synsets = swn.senti_synsets(next_word)
			
			if len(curr_senti_synsets) > 0 and len(next_senti_synsets) > 0:
				c_pos_curr = curr_senti_synsets[0].pos_score()
				c_pos_next = next_senti_synsets[0].pos_score()
				sent = (c_pos_curr - c_pos_next)
				if sent != 0:
					sent_list.append(sent)
		if len(sent_list) > MAX_LEN:
			sent_list = sent_list[:MAX_LEN]
		for i in range(MAX_LEN - len(sent_list)):
			sent_list.append(0.0)
		inp_X.append(sent_list)
	return inp_X

def get_sent_neg_comp(corpus):
	for curr_sentence in corpus:
		curr_sentence = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', curr_sentence)
		curr_sentence = re.sub(r'@[a-zA-Z0-9_]+', ' ', curr_sentence)
		sentence_tokenized = tokenizer(curr_sentence)
		sent_list = []
		for i in range(len(sentence_tokenized)-1):
			curr_word = sentence_tokenized[i]
			next_word = sentence_tokenized[i+1]
			if curr_word.startswith('#'):
				curr_word = curr_word[1:]
			if next_word.startswith('#'):
				next_word = next_word[1:]

			curr_senti_synsets = swn.senti_synsets(curr_word)
			next_senti_synsets = swn.senti_synsets(next_word)
			
			if len(curr_senti_synsets) > 0 and len(next_senti_synsets) > 0:
				c_pos_curr = curr_senti_synsets[0].neg_score()
				c_pos_next = next_senti_synsets[0].neg_score()
				sent = (c_pos_curr - c_pos_next)
				if sent != 0:
					sent_list.append(sent)
		if len(sent_list) > MAX_LEN:
			sent_list = sent_list[:MAX_LEN]
		for i in range(MAX_LEN - len(sent_list)):
			sent_list.append(0.0)
		inp_X.append(sent_list)
	return inp_X

def get_sent_obj_comp(corpus):
	for curr_sentence in corpus:
		curr_sentence = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', curr_sentence)
		curr_sentence = re.sub(r'@[a-zA-Z0-9_]+', ' ', curr_sentence)
		sentence_tokenized = tokenizer(curr_sentence)
		sent_list = []
		for i in range(len(sentence_tokenized)-1):
			curr_word = sentence_tokenized[i]
			next_word = sentence_tokenized[i+1]
			if curr_word.startswith('#'):
				curr_word = curr_word[1:]
			if next_word.startswith('#'):
				next_word = next_word[1:]

			curr_senti_synsets = swn.senti_synsets(curr_word)
			next_senti_synsets = swn.senti_synsets(next_word)
			
			if len(curr_senti_synsets) > 0 and len(next_senti_synsets) > 0:
				c_pos_curr = curr_senti_synsets[0].obj_score()
				c_pos_next = next_senti_synsets[0].obj_score()
				sent = (c_pos_curr - c_pos_next)
				if sent != 0:
					sent_list.append(sent)
		if len(sent_list) > MAX_LEN:
			sent_list = sent_list[:MAX_LEN]
		for i in range(MAX_LEN - len(sent_list)):
			sent_list.append(0.0)
		inp_X.append(sent_list)
	return inp_X

def get_sent_using_twitter_extractor(corpus):
	for curr_sentence in corpus:
		print curr_sentence
		sentence_tokenized = []
		sentence_tokenized1 = keyword_extraction_w_parser.get_keywords(curr_sentence)
		sentence_tokenized2 = keyword_extraction_w_parser.extract_hashtag(curr_sentence)
		if len(sentence_tokenized2) > 0:
			sentence_tokenized = sentence_tokenized1.append(sentence_tokenized2)
		else:
			sentence_tokenized = sentence_tokenized1
		print sentence_tokenized
		sent_list = []
		for i in range(len(sentence_tokenized)-1):
			curr_word = sentence_tokenized[i]
			next_word = sentence_tokenized[i+1]

			curr_senti_synsets = swn.senti_synsets(curr_word)
			next_senti_synsets = swn.senti_synsets(next_word)
			
			if len(curr_senti_synsets) > 0 and len(next_senti_synsets) > 0:
				c_pos_curr = curr_senti_synsets[0].pos_score()
				c_pos_next = next_senti_synsets[0].pos_score()
				sent = (c_pos_curr - c_pos_next)
				if sent != 0:
					sent_list.append(sent)
		if len(sent_list) > MAX_LEN:
			sent_list = sent_list[:MAX_LEN]
		for i in range(MAX_LEN - len(sent_list)):
			sent_list.append(0.0)
		inp_X.append(sent_list)
	return inp_X

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
		if len(sent_list) > MAX_LEN:
			sent_list = sent_list[:MAX_LEN]
		for i in range(MAX_LEN - len(sent_list)):
			sent_list.append(0.0)
		inp_X.append(sent_list)
	return inp_X

inp_X = get_sent_max_pos_neg(corpus)
csv = open("test_csv.csv", "w")
for i in range(len(inp_X)):
	myList = ','.join(map(str, inp_X[i]))
	row = myList + "," + str(inp_y[i]) + "\n"
	csv.write(row)

dataset = []
# inp_X = get_sent_pos_comp(corpus)
for i in range(len(inp_X)):
	tmp = inp_X[i]
	tmp.append(inp_y[i])
	dataset.append(tmp)
dataset = np.array(dataset)
print dataset.shape


X = dataset[:,0:10]
Y = dataset[:,10]
seed = 7
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=test_size, random_state=seed)
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

