from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from itertools import chain
from nltk.corpus import wordnet

def write_file(name, data):
    with open(name, 'w') as fi:
        for line in data:
            fi.write(line)
            fi.close()

def extrac_import_feature(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if feature_names is not None:
            print("---top 10 keywords per class:")
            for i, label in enumerate(target_names):
                #print (len(np.argsort(clf.coef_[i])))
                count = int(len(np.argsort(clf.coef_[i])) * 0.1)
                print(count)
                top10 = np.argsort(clf.coef_[i])[-count:]
                #print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
                print("%s: %s" % (label, " ".join(feature_names[top10])))
        print()


    print("classification report:")
    print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

if __name__ == '__main__':
    ##
    data_train = load_files('train')
    data_test = load_files('test')
    print('data loaded')

    # order of labels in `target_names` can be different from `categories`
    target_names = data_train.target_names

    # split a training set and a test set
    y_train, y_test = data_train.target, data_test.target

    print("Extracting features from the training data using a sparse vectorizer")
    t0 = time()
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
    duration = time() - t0
    # print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()

    print("Extracting features from the test data using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(data_test.data)
    duration = time() - t0
    # print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()

    # mapping from integer feature name to original token string
    feature_names = vectorizer.get_feature_names()

    if feature_names:
        feature_names = np.asarray(feature_names)

    # results values
    results = []

    ##
    # Train sparse Naive Bayes classifiers
    print('=' * 80)
    print("Naive Bayes")
    results.append(extrac_import_feature(MultinomialNB(alpha=.01)))

    # wordnet
    synonyms = wordnet.synsets('change')
    lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
    print(lemmas)