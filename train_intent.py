
from __future__ import print_function

import logging
from _operator import contains
import os
import numpy as np
from optparse import OptionParser
import sys
from time import time
import csv
import matplotlib.pyplot as plt

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

if __name__ == '__main__':

    data_train = load_files('data/train')
    data_test = load_files('data/test')
    print('data loaded')
    list_label = os.listdir('./dict')
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data_train.data)
    X_train_counts.shape
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_train_tf.shape
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf.shape
    clf = MultinomialNB().fit(X_train_tfidf, data_train.target)

    # predict
    docs_new = ['how to break login my system', 'No, I don’t remember my ID']
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)
    for doc, category in zip(docs_new, predicted):
        label = data_train.target_names[category]
        print('No match dict')
        print('%r => %s' % (doc, label))
        path = './dict/' + label
        file_content = open(path, 'r')
        #print(file_content.read())
        file_content = file_content.read()
        content = file_content.split()
        docs = doc.split()
        #print(file_content)
        flg = 0
        print('Match dict')
        for texts in docs:
            if texts in content:
               flg = 1
        if flg == 1:
            print('%r => %s' % (doc, label))
        else:
            print('%r => %s' % (doc, 'unknown'))

