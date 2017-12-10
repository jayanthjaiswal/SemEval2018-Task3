#!/usr/bin/env python3

from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import dump_svmlight_file
from sklearn import metrics
import numpy as np
import logging
import codecs
import re
from nltk.corpus import sentiwordnet as swn

# Import MNIST data

import tensorflow as tf

# Parameters
learning_rate = 0.1
num_steps = 1000
batch_size = 64
display_step = 10

# Network Parameters
n_hidden_1 = 11  # 1st layer number of neurons
n_hidden_2 = 6  # 2nd layer number of neurons
num_input = 15  # MNIST data input (img shape: 28*28)
num_classes = 2  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

logging.basicConfig(level=logging.INFO)

MAX_LEN = 15
inp_y = []
one_hot_y = []
inp_X = []


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
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]
                y.append(label)
                corpus.append(tweet)

    return corpus, y


def featurize(corpus):
    '''
    Tokenizes and creates TF-IDF BoW vectors.
    :param corpus: A list of strings each string representing document.
    :return: X: A sparse csr matrix of TFIDF-weigted ngram counts.
    '''

    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words="english")
    X = vectorizer.fit_transform(corpus)
    # print(vectorizer.get_feature_names()) # to manually check if the tokens are reasonable
    return X


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def gen(random_perm, batch_size):
    global inp_X, inp_y, one_hot_y
    inp_perm_X = [inp_X[i] for i in random_perm]
    inp_perm_Y = [one_hot_y[i] for i in random_perm]
    batch_start = 0
    num_examples = len(inp_X)
    print('Number of examples: ' + str(num_examples))
    while 1:
        temp_X = inp_perm_X[batch_start:min(num_examples, batch_start + batch_size)][:]
        temp_Y = inp_perm_Y[batch_start:min(num_examples, batch_start + batch_size)][:]
        batch_start = (min(num_examples, batch_start + batch_size))
        if batch_start == batch_size:
            batch_start = 0
        ret_X = np.asarray(temp_X)
        ret_Y = np.asarray(temp_Y)
        yield ret_X, ret_Y


if __name__ == "__main__":
    # Experiment settings

    global inp_X, inp_y, one_hot_y

    # Dataset: SemEval2018-T4-train-taskA.txt or SemEval2018-T4-train-taskB.txt
    DATASET_FP = "./SemEval2018-T4-train-taskA.txt"
    TASK = "A"  # Define, A or B
    FNAME = './predictions-task' + TASK + '.txt'
    PREDICTIONSFILE = open(FNAME, "w")

    K_FOLDS = 10  # 10-fold crossvalidation
    CLF = LinearSVC()  # the default, non-parameter optimized linear-kernel SVM

    # Loading dataset and featurised simple Tfidf-BoW model
    corpus, inp_y = parse_dataset(DATASET_FP)
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize

    inp_X = []

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
    permute = np.random.permutation(len(inp_X));
    # print (X.size, X.shape)
    # X = featurize(corpus)

    one_hot_y = [(1, 0) if i == 1 else (0, 1) for i in inp_y]

    # Remember to correct X to inp_X and y to inp_y
    # print(y.shape)
    # print(y)
    # print(len(X), len(X[0]))

    # class_counts = np.asarray(np.unique(y, return_counts=True)).T.tolist()
    # print (class_counts)
    #
    # # Returns an array of the same size as 'y' where each entry is a prediction obtained by cross validated
    # predicted = cross_val_predict(CLF, X, y, cv=K_FOLDS)
    #
    # # Modify F1-score calculation depending on the task
    # if TASK.lower() == 'a':
    #     score = metrics.f1_score(y, predicted, pos_label=1)
    # elif TASK.lower() == 'b':
    #     score = metrics.f1_score(y, predicted, average="macro")
    # print ("F1-score Task", TASK, score)
    # for p in predicted:
    #     PREDICTIONSFILE.write("{}\n".format(p))
    # PREDICTIONSFILE.close()

    # Construct model
    logits = neural_net(X)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    generator = gen(permute, batch_size);
    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for step in range(1, num_steps + 1):
            batch_x, batch_y = generator.next()
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
                print(batch_x.shape);

        print("Optimization Finished!")

