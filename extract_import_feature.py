from __future__ import print_function

import logging
from _operator import contains

import numpy as np
from optparse import OptionParser
import sys
from time import time
import csv
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
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, map_tag
import nltk

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

    ##value
    list_label = []
    list_list_feature = []

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
            print("---top keywords per class:")
            for i, label in enumerate(target_names):
                #print (len(np.argsort(clf.coef_[i])))
                count = int(len(np.argsort(clf.coef_[i])) * 0.1)
                print(count)
                top10 = np.argsort(clf.coef_[i])[-count:]
                #print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
                print("%s: %s" % (label, " ".join(feature_names[top10])))
                list_feature = list(feature_names[top10])
                list_list_feature.append(list_feature)
                list_label.append(label)

        print()


    #print("classification report:")
    #print(metrics.classification_report(y_test, pred,
                                            #target_names=target_names))

    print()
    clf_descr = str(clf).split('(')[0]
    #return clf_descr, score, train_time, test_time
    return list_label, list_list_feature

def process_csv():
    category, content = [], []
    with open('1e6aa914555236a0c9cea7a330d5a04e_170607144459.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            category.append(row['intent'])
            content.append(row['text'])
    return category, content

def contains_word(s, w):
    return (' ' + w + ' ') in (' ' + s + ' ')

def find_pos_word(label, text, intent, content):
    #
    list_pos = []
    for id in range(len(intent)):
        if label == intent[id]:
            if contains_word(content[id], text):
                tokenize = nltk.word_tokenize(content[id])
                sentence_pos = pos_tag(word_tokenize(content[id]))
                simplifiedTags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in sentence_pos]
                for token in simplifiedTags:
                    if text == token[0] and not token[1] in list_pos:
                        list_pos.append(token[1])
    return list_pos

def list_synonyms(text_p, text):
    #for text_p in lis_pos:
        if text_p == 'ADJ':
            synonyms = wordnet.synsets(text, pos=wordnet.ADJ)
            return synonyms
        elif text_p == 'ADV':
            synonyms = wordnet.synsets(text, pos=wordnet.ADV)
            return synonyms
        elif text_p == 'CONJ':
            synonyms = wordnet.synsets(text, pos=wordnet.CONJ)
            return synonyms
        elif text_p == 'DET':
            synonyms = wordnet.synsets(text, pos=wordnet.DET)
            return synonyms
        elif text_p == 'NOUN':
            synonyms = wordnet.synsets(text, pos=wordnet.NOUN)
            return synonyms
        elif text_p == 'NUM':
            synonyms = wordnet.synsets(text, pos=wordnet.NUM)
            return synonyms
        elif text_p == 'PRT':
            synonyms = wordnet.synsets(text, pos=wordnet.PRT)
            return synonyms
        elif text_p == 'PRON':
            synonyms = wordnet.synsets(text, pos=wordnet.PRON)
            return synonyms
        elif text_p == 'VERB':
            synonyms = wordnet.synsets(text, pos=wordnet.VERB)
            return synonyms
        else:
            return []

if __name__ == '__main__':

    ##
    data_train = load_files('data/train')
    data_test = load_files('data/test')
    print('data loaded')
    ## add
    intent, content = process_csv()
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
    #results = []

    # Train sparse Naive Bayes classifiers
    print('=' * 80)
    print("Naive Bayes")
    #results.append(extrac_import_feature(MultinomialNB(alpha=.01)))
    list_lab, list_list_fe = extrac_import_feature(MultinomialNB(alpha=.01))

    # seach wordnet
    for ind in range(len(list_list_fe)):
        label_i = list_lab[ind]
        collec = []
        for text in list_list_fe[ind]:
            lis_pos = find_pos_word(label_i,text, intent, content)
            ## pos -> find
            for text_pos in lis_pos:
                synonyms = list_synonyms(text_pos, text)
                #if synonyms:
                lemmas = list(chain.from_iterable([word.lemma_names() for word in synonyms]))
                lemmas.append(text)
                collec = collec + lemmas
                print('label: ' + text )
                print(lemmas)
            # synonyms = wordnet.synsets(text, pos=wordnet.ADV)
            # lemmas = list(chain.from_iterable([word.lemma_names() for word in synonyms]))
            # lemmas.append(text)
            # collec = collec + lemmas
            # #print(lemmas)
        collec = set(collec)
        print(collec)
        print('*' * 80)

        ## write file
        data_feature = " ".join(collec)
        write_file(label_i,data_feature)
