** The xgd_boost_test module caan be downloaded and run independently of all other code in this repo **

## There are two main scripts in this directory that are of importance: ##
1. boost_test.py 
2. boost_test_2.py 

## To run this code ##
1. download this directory
2. cd to the directory and run either:
	python boost_test.py
	python boost_test_2.py

## boost_test.py ##

This file shows how sentiment scores were extracted from the tweets data and preprocessed. The next step was to pass the data through the xgboost python library to train a model. The model was then tested on a test set and the accuracies were reported.

Some examples of feature extraction were extracting positive, negative, and objective scores independently. Other examples include combining these scores in different ways to get optimal performance.


## boost_test_2.py ##

This file shows how word embeddings and sentiment scores were combined and preprocessed. The next step was to pass the data through the xgboost python library to train a model. Then the model was tested on a test set and the accuracies were reported.
