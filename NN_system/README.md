## Description ##

There are two main scripts in this directory that are of importance: 
1. train_sent.py  (using sentiment scores for CNN)
2. train_tfidf_sent.py (using sentiment scores and tfidf for CNN)

## Evaluation ##
To run this code:
1. download this directory
2. cd to the directory and run either:
	python train_sent.py
	python train_tfidf_sent.py

## train_sent.py ##

This file shows how sentiment scores were extracted from the tweets data and preprocessed. The next step was to pass the data through the CNN, implemented with Tensorflow python library to train a model and to report the accuracy.

## train_tfidf_sent.py ##

This file shows how sentiment scores and tfidf were extracted from the tweets data and preprocessed. The next step was to pass the data through the CNN, implemented with Tensorflow python library to train a model and to report the accuracy.
