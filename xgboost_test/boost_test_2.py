from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import re
import numpy as np
import os

#preprocess data files that contain word_embedding features.
#May need to change path based on where data is
data_y_fp = "./labels.txt"
directory = "word_embeddings"
MAX_LENGTH = 310
data_x = []
data_y = []

#go through labels file:
with open(data_y_fp) as f:
	for label in f:
		label = int(label)
		data_y.append(label)
data_y = np.array(data_y)

#go through the feature files. Each file has 25 features
files = os.listdir(directory)
os.chdir(directory)
for filename in files:
    example = []
    name = filename.split(".")
    file_num = int(name[0])
    label = data_y[file_num-1]
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
    example = np.append(example, label)
    data_x.append(example)
data_x = np.array(data_x)
X = data_x[:,0:MAX_LENGTH]
Y = data_x[:,MAX_LENGTH]

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

