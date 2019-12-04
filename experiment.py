# from flask import Flask, render_template, url_for, request
import csv
# from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import math
import time
from nltk.corpus import stopwords
# import spacy
from textblob import Word

start_time = time.time()
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


# app = Flask(__name__)
readcsv = []
description = {}
freq_dictonary = {}
term_freq = {}
tf_idf = {}
search_result = {}
key_names = {}
tf_idf_full = {}
search_description = {}
tfidf_display = {}
synonyms = []
text = ''
total_result = 0
vocab_spacy = {}
vocab_wordnet = {}
vocab_textblob = {}


stop_words = set(stopwords.words('english'))
# nlp = spacy.load("en_core_web_sm")
with open('data.csv', encoding="utf8") as csvfile:
    readcsv = csv.reader(csvfile, delimiter=',')
    for row in readcsv:
        description[row[32]] = row[6]
        key_names[row[32]] = row[17]
        word_tokens = nltk.word_tokenize(row[6])
        filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words]
        search_description[row[32]] = ''.join(filtered_sentence)
        lemmatizer = WordNetLemmatizer()

        for w in word_tokens:
            # temp = lemmatizer.lemmatize(w)
            # if w in vocab_wordnet:
            #     vocab_wordnet[w] +=1
            # else:
            #     vocab_wordnet[w] = 1
            temp = w.lemma_
            if temp in vocab_spacy:
                vocab_spacy[w] += 1
            else:
                vocab_spacy[w] = 1
            # u = Word(w)
            # temp = u.lemmatize()
            # if temp in vocab_textblob:
            #     vocab_textblob[temp] += 1
            # else:
            #     vocab_textblob[temp] = 1
print("length of wordnet:",len(vocab_wordnet)," length of textblob:",len(vocab_textblob))


