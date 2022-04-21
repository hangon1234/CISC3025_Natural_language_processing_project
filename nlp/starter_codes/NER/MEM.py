#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# --------------------------------------------------
# Description:
# --------------------------------------------------
# Author: Konfido <konfido.du@outlook.com>
# Created Date : April 4th 2020, 17:45:05
# Last Modified: April 4th 2020, 17:45:05
# --------------------------------------------------
import nltk
nltk.download('names')
nltk.download('brown')
nltk.download('punkt')

from nltk.classify.maxent import MaxentClassifier
from sklearn.metrics import (accuracy_score, fbeta_score, precision_score,
                             recall_score)
import os
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import names, brown

NAMES = set(names.words())
BROWN = set(brown.words())

class MEMM():
    def __init__(self):
        self.train_path = "./starter_codes/data/train"
        self.dev_path = "./starter_codes/data/dev"
        self.beta = 0
        self.max_iter = 0
        self.classifier = None

    def features(self, words, previous_label, position):
        """
        Note: The previous label of current word is the only visible label.

        :param words: a list of the words in the entire corpus
        :param previous_label: the label for position-1 (or O if it's the start
                of a new sentence)
        :param position: the word you are adding features for
        """

        features = {}
        """ Baseline Features """
        current_word = words[position]
        features[f'has_{current_word}'] = 1
        features['prev_label'] = previous_label
        if current_word[0].isupper():
            features['Titlecase'] = 1

        # Baseline score
        # f_score = 0.8715
        # accuracy = 0.9641
        # recall = 0.7143
        # precision = 0.9642


        #===== TODO: Add your features here =======#
        # When whole word is capitalized
        if current_word.isupper():
            features['current_allcap'] = 1

        # if current_word.islower():
        #     features['current_lower'] = 1

        # Using NLTK Names corpus
        if current_word in NAMES:
            features["name"] = 1

        # Previous word
        # if position > 0:
        #     features[f"prev_word_{words[position-1]}"] = 1

        # If camelcase
        # if [x.isupper for x in current_word].count(True) > 1:
        #     features["camel_case"] = 1

        # if contains hyphen
        # if '-' in current_word:
        #     features["hyphen"] = 1

        # if underscore
        # if '_' in current_word:
        #     features["underscore"] = 1

        # Previous two word
        # if position > 1:
        #     features[f"prev_two_word{words[position-2]}"] = 1

        # Next word
        # if position + 1 != len(words):
        #     features[f"next_word_{words[position+1]}"] = 1

        #=============== TODO: Done ================#
        return features

    def load_data(self, filename):
        words = []
        labels = []
        for line in open(filename, "r", encoding="utf-8"):
            doublet = line.strip().split("\t")
            if len(doublet) < 2:     # remove emtpy lines
                continue
            words.append(doublet[0])
            labels.append(doublet[1])
        return words, labels

    def train(self):
        print('Training classifier...')
        words, labels = self.load_data(self.train_path)
        previous_labels = ["O"] + labels
        features = [self.features(words, previous_labels[i], i)
                    for i in range(len(words))]
        train_samples = [(f, l) for (f, l) in zip(features, labels)]
        classifier = MaxentClassifier.train(
            train_samples, max_iter=self.max_iter)
        self.classifier = classifier

    def test(self):
        print('Testing classifier...')
        words, labels = self.load_data(self.dev_path)
        previous_labels = ["O"] + labels
        features = [self.features(words, previous_labels[i], i)
                    for i in range(len(words))]
        results = [self.classifier.classify(n) for n in features]

        f_score = fbeta_score(labels, results, average='macro', beta=self.beta)
        precision = precision_score(labels, results, average='macro')
        recall = recall_score(labels, results, average='macro')
        accuracy = accuracy_score(labels, results)

        print("%-15s %.4f\n%-15s %.4f\n%-15s %.4f\n%-15s %.4f\n" %
              ("f_score=", f_score, "accuracy=", accuracy, "recall=", recall,
               "precision=", precision))

        return True

    def predict_sentence(self, sentence):
        self.load_model()
        # Tokenize the sentence
        words = word_tokenize(sentence)
        # Insert some dummy label
        labels = ["O"] * len(words)
        # Add one more dummy label to extract features
        previous_labels = ["O"] + labels
        # Extract features for given sentence.
        features = [self.features(words, previous_labels[i], i)
                    for i in range(len(words))]
        # Get classification result (result labels)
        results = [self.classifier.classify(n) for n in features]
        # Return words and classification result
        return words, results

    def show_samples(self, bound):
        """Show some sample probability distributions.
        """
        words, labels = self.load_data(self.train_path)
        previous_labels = ["O"] + labels
        features = [self.features(words, previous_labels[i], i)
                    for i in range(len(words))]
        (m, n) = bound
        pdists = self.classifier.prob_classify_many(features[m:n])

        print('  Words          P(PERSON)  P(O)\n' + '-' * 40)
        for (word, label, pdist) in list(zip(words, labels, pdists))[m:n]:
            if label == 'PERSON':
                fmt = '  %-15s *%6.4f   %6.4f'
            else:
                fmt = '  %-15s  %6.4f  *%6.4f'
            print(fmt % (word, pdist.prob('PERSON'), pdist.prob('O')))

    def dump_model(self):
        with open('./starter_codes/model.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)

    def load_model(self):
        with open('starter_codes/model.pkl', 'rb') as f:
            self.classifier = pickle.load(f)
