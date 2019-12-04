import os
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, render_template, url_for, request
import csv
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import math
import time
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from decimal import *
import pandas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score,recall_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.tests.test_pprint import SimpleImputer
import numpy as np

start_time = time.time()
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

app = Flask(__name__)
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
class_dict = defaultdict(list)
class_probability = {}
vocabulary = {}
class1words = {}
class2words = {}
class3words = {}
class4words = {}
class5words = {}
total_count_words = {}
vocab_count = 0
den_val = {}
classification = {}
term_documents = {}
class_rec = {}
search_result_classify = {}
class1count = 0
class2count = 0
class3count = 0
class4count = 0
class5count = 0
p_word_class1 = 0
p_word_class2 = 0
p_word_class3 = 0
p_word_class4 = 0
p_word_class5 = 0
image_file_dict = {}
list_to_append = {}
t = {}
p = {}
special_characters = [',', '.', '?', '!']
captions = {}
URLs = {}
image_description ={}
captions_display = {}
phrase_search = {}

@app.route('/')
def index():
    readfile()
    # return render_template('index.html', result=search_result, tfidf_display=tfidf_display, text=text)
    return render_template('home.html')


@app.route("/classify_page",methods=['POST', 'GET'])
def classify_page():
    search_result.clear()
    return render_template('classifier.html', result=search_result_classify, text=text)


@app.route("/search_page", methods=['POST', 'GET'])
def search_page():
    search_result.clear()
    text=""
    total_result = 0
    tfidf_display.clear()
    return render_template('index.html', result=search_result, tfidf_display=tfidf_display, text=text,total_result=total_result)


@app.route("/search", methods=['POST', 'GET'])
def search():
    global f1, l1, l2, l2_split, l1_split
    if request.method == 'POST':
        text = request.form['Search text']
        tf_idf_full.clear()
        tfidf_display.clear()
        if text:
            string = text.split(" ")
            if string[0] == "synonyms:":
                for i in range(1, len(string)):
                    l=0
                    for syn in wordnet.synsets(str(string[i])):
                        for l in syn.lemmas():
                            if str(l.name()) in string:
                                pass
                            else:
                                string.append(str(l.name()))
                for i in range(1, len(string)):

                    for key, value in search_description.items():
                        if str(string[i]).lower() in value:

                            computetf(string[i])

                            idf = computeidf()
                            tf_idf.clear()
                            for key1, value1 in term_freq.items():
                                tf_idf[key1] = computetfidf(term_freq[key1], idf)
                    for key2, value2 in tf_idf.items():
                        if key2:
                            if key2 in tf_idf_full:
                                tf_idf_full[key2] = tf_idf_full[key2] + tf_idf[key2]
                            else:
                                tf_idf_full[key2] = tf_idf[key2]
                            tfidf_display[key_names[key2]] = tf_idf_full[key2]
                display(text)
                total_result = len(tfidf_display)
                print("--- %s seconds ---" % (time.time() - start_time))

                t = ""
                for s in string:
                    t = t + " " + s
                return render_template('index.html', result=search_result, tfidf_display=tfidf_display, text=t, total_result=total_result)
            elif string[0] == "lemmatize:":
                for i in range(1, len(string)):
                    tf_idf.clear()
                    lemmatizer = WordNetLemmatizer()
                    string[i] = lemmatizer.lemmatize(string[i])

                    for key, value in search_description.items():
                        if string[i] in value:
                            computetf(string[i])
                            idf = computeidf()
                            for key1, value1 in term_freq.items():
                                tf_idf[key1] = computetfidf(term_freq[key1], idf)
                    for key2, value2 in tf_idf.items():
                        if key2 in tf_idf_full:
                            tf_idf_full[key2] = tf_idf_full[key2] + tf_idf[key2]
                        else:
                            tf_idf_full[key2] = tf_idf[key2]
                        tfidf_display[key_names[key2]] = tf_idf_full[key2]
                display(text)
                print("--- %s seconds ---" % (time.time() - start_time))

                t = ""
                for s in string:
                    t = t + " " + s
                total_result = len(tfidf_display)
                return render_template('index.html', result=search_result, tfidf_display=tfidf_display, text=t, total_result=total_result)
            elif "\"" in str(text):
                str_phrase = str(text).replace("\"", "")
                stop_words = set(stopwords.words('english'))
                word_tokens = nltk.word_tokenize(str(str_phrase))
                if len(word_tokens) > 1:
                    f1 = word_tokens[0]
                    if f1 in vocabulary:
                        l1 = vocabulary[f1]
                        l1_split=l1.split(',')
                    f2 = word_tokens[1]
                    if f2 in vocabulary:
                        l2= vocabulary[f2]
                        l2_split= l2.split(',')
                        print(l2_split)
                    if (f1 in vocabulary) and (f2 in vocabulary):
                        for i in range(len(l1_split)):
                            if len(l1_split[i]) < 5:
                                continue
                            for j in range(len(l2_split)):
                                if len(l2_split[j]) < 5:
                                    continue
                                if l1_split[i] == l2_split[j]:
                                    if (int(l1_split[i+1]) == int(l2_split[j+1]) - 1) or (int(l1_split[i+1]) == int(l2_split[j+1]) + 1):

                                        if l2_split[j] in phrase_search:
                                            phrase_search[l2_split[j]] +=1
                                        else:
                                            phrase_search[l2_split[j]]= 1
                string = str_phrase
                string = string.split(" ")
                for i in range(len(string)):
                    tf_idf.clear()
                    for key, value in search_description.items():
                        if str(string[i]).lower() in value:
                            computetf(string[i])
                            idf = computeidf()
                            for key1, value1 in term_freq.items():
                                tf_idf[key1] = computetfidf(term_freq[key1], idf)
                    for key2, value2 in tf_idf.items():
                        if key2 in tf_idf_full:
                            tf_idf_full[key2] = tf_idf_full[key2] + tf_idf[key2]
                        else:
                            tf_idf_full[key2] = tf_idf[key2]
                        tfidf_display[key_names[key2]] = tf_idf_full[key2]
                search_result.clear()
                for keys in sorted(tf_idf_full, key=tf_idf_full.get, reverse=True):
                    hotel_name = key_names[keys]
                    if hotel_name not in search_result:
                        if keys in phrase_search:
                                search_result[hotel_name] = description[keys]
                total_result = len(phrase_search)
                return render_template('index.html', result=search_result, tfidf_display=tfidf_display, text=text, total_result=total_result)
            else:
                for i in range(len(string)):
                    tf_idf.clear()
                    for key, value in search_description.items():
                        if str(string[i]).lower() in value:
                            computetf(string[i])
                            idf = computeidf()
                            for key1, value1 in term_freq.items():
                                tf_idf[key1] = computetfidf(term_freq[key1], idf)
                    for key2, value2 in tf_idf.items():
                        if key2 in tf_idf_full:
                            tf_idf_full[key2] = tf_idf_full[key2] + tf_idf[key2]
                        else:
                            tf_idf_full[key2] = tf_idf[key2]
                        tfidf_display[key_names[key2]] = tf_idf_full[key2]
                display(text)
                total_result = len(tfidf_display)
                return render_template('index.html', result=search_result, tfidf_display=tfidf_display, text=text, total_result=total_result)
        else:
            term_freq.clear()
            tf_idf.clear()
            tf_idf_full.clear()
            tfidf_display.clear()
            search_result.clear()
            total_result = 0
            return render_template('index.html', result=search_result, tfidf_display=tfidf_display, text=text, total_result=total_result)
    else:
        return render_template('index.html')


@app.route("/classify", methods=['POST', 'GET'])
def classify():
    global class1count,class2count,class3count,class4count,class5count
    # vid = pandas.read_csv("data.csv")
    # x_train, x_test, y_train, y_test= train_test_split(vid['hotel_description'], vid['hotel_star_rating'], test_size=0.20, random_state=1, shuffle=False)
    # count_vector = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',lowercase=True, stop_words='english')
    # x_train_cv = count_vector.fit_transform(x_train.values.astype('U'))
    # x_test_cv = count_vector.transform(x_test.values.astype('U'))
    # nb=MultinomialNB(alpha=1)
    # nb.fit(x_train_cv, y_train)
    # predictions= nb.predict(x_test_cv)
    # # rms = sklearn.metrics.mean_squared_error(y_test, predictions)
    # # print("rms: ", rms)
    #
    # SVM = svm.SVC(C=1.0,kernel='linear',degree=3,gamma='auto')
    # SVM.fit(x_train_cv,y_train)
    # predictions_svm=SVM.predict(x_test_cv)
    #
    # kdt = KNeighborsClassifier(n_neighbors=3)
    # kdt.fit(x_train_cv,y_train)
    # predictions_knn = kdt.predict(x_test_cv)
    #
    # clf = DecisionTreeClassifier()
    # clf = clf.fit(x_train_cv, y_train)
    # y_pred = clf.predict(x_test_cv)
    #
    # print('accuracy of naive: ', (accuracy_score(y_test, predictions))*100)
    # print('accuracy of svm:', (accuracy_score(y_test, predictions_svm))*100)
    # print('accuracy of knn:', (accuracy_score(y_test, predictions_knn)) * 100)
    # print('accuracy of decision tree:', (accuracy_score(y_test, y_pred)) * 100)
    text = request.form['Classify text']
    string = text.split(" ")
    p_c1=1
    p_c2=1
    p_c3=1
    p_c4=1
    p_c5=1
    getcontext().prec = 5
    for s in string:
        p_word_class1 = 0
        p_word_class2 = 0
        p_word_class3 = 0
        p_word_class4 = 0
        p_word_class5 = 0
        class1count = 1
        class2count = 1
        class3count = 1
        class4count = 1
        class5count = 1
        for key, value in search_description.items():
            if str(s).lower() in value:
                term_documents[key] = 1
                x = class_rec[key]
                if x == "1 Star hotel":
                    class1count +=1
                elif x == "2 Star hotel":
                    class2count +=1
                elif x == "3 Star hotel":
                    class3count +=1
                elif x == "4 Star hotel":
                    class4count +=1
                elif x == "5 Star hotel":
                    class5count +=1

        p_word_class1 = class1count/len(class_dict["1 Star hotel"])
        p_word_class2 = class2count/len(class_dict["2 Star hotel"])
        p_word_class3 = class3count / len(class_dict["3 Star hotel"])
        p_word_class4 = class4count / len(class_dict["4 Star hotel"])
        p_word_class5 = class5count / len(class_dict["5 Star hotel"])

        p_c1 = p_c1 * p_word_class1
        p_c2 = p_c2 * p_word_class2
        p_c3 = p_c3 * p_word_class3
        p_c4 = p_c4 * p_word_class4
        p_c5 = p_c5 * p_word_class5

    p_c1 = p_c1 * len(class_dict["1 Star hotel"])/5000
    p_c2 = p_c2 * len(class_dict["2 Star hotel"]) / 5000
    p_c3 = p_c3 * len(class_dict["3 Star hotel"]) / 5000
    p_c4 = p_c4 * len(class_dict["4 Star hotel"]) / 5000
    p_c5 = p_c5 * len(class_dict["5 Star hotel"]) / 5000
    search_result_classify.clear()
    sum = p_c1+p_c2+p_c3+p_c4+p_c5
    search_result_classify["1 Star hotel"] = p_c1*100/sum
    search_result_classify["2 Star hotel"] = p_c2*100/sum
    search_result_classify["3 Star hotel"] = p_c3*100/sum
    search_result_classify["4 Star hotel"] = p_c4*100/sum
    search_result_classify["5 Star hotel"] = p_c5*100/sum

    return render_template('classifier.html', result=search_result_classify, text=text)


@app.route("/Image_Search", methods=['POST', 'GET'])
def image_page():

    search_result.clear()
    stop_words = set(stopwords.words('english'))
    with open('image.csv', encoding="utf8") as csvfile:
        readcsv = csv.reader(csvfile, delimiter=',')
        for row in readcsv:
            captions[row[0]] = row[2]
            URLs[row[0]] = row[1]
            word_tokens = nltk.word_tokenize(row[2])
            filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words]
            image_description[row[0]] = ''.join(filtered_sentence)
    csvfile.close()
    image_file_dict.clear()
    return render_template('Image_Search.html', image_file_dict = image_file_dict, text=text)


@app.route("/image", methods=['POST','GEt'])
def image_search():
    # kmm
    if request.method == 'POST':
        text = request.form['Search text']
        tf_idf_full.clear()
        tfidf_display.clear()
        image_file_dict.clear()
        captions_display.clear()
        term_freq.clear()
        if text:
            string = text.split(" ")
            for i in range(len(string)):
                tf_idf.clear()
                for key, value in image_description.items():
                    if str(string[i]).lower() in value:
                        for key1, value1 in image_description.items():
                            if str(string[i]).lower() in value1:
                                desc_str = image_description[key1]
                                term_freq[key1] = desc_str.count(str(string[i]).lower())
                        if len(term_freq):
                            idf= 1 + math.log2(len(captions) / len(term_freq))
                        else:
                            idf= 0

                        for key2, value2 in term_freq.items():
                            tf_idf[key2] = computetfidf(term_freq[key2], idf)

                for key3, value3 in tf_idf.items():
                    if key3 in tf_idf_full:
                        tf_idf_full[key3] = tf_idf_full[key3] + tf_idf[key3]
                    else:
                        tf_idf_full[key3] = tf_idf[key3]
                    tfidf_display[key3] = tf_idf_full[key3]
                    image_file_dict[key3] = URLs[key3]
                    captions_display[key3] = captions[key3]
            total_result = len(tfidf_display)
        else:
            captions_display.clear()
            tfidf_display.clear()
            text = ""
            image_file_dict.clear()
            total_result = 0
            tf_idf_full.clear()
            term_freq.clear()
            string =""
    return render_template('Image_Search.html', image_file_dict = image_file_dict, tfidf_display=tfidf_display,captions_display=captions_display, text=text,total_result=total_result)


def display(text):
    search_result.clear()
    for keys in sorted(tf_idf_full, key=tf_idf_full.get, reverse=True):
        hotel_name = key_names[keys]
        if hotel_name not in search_result:
            search_result[hotel_name] = description[keys]


def countwords(t):
    word = t.split(" ")
    return len(word)


def computetf(str1):
    for key, value in search_description.items():
        if str(str1).lower() in value:
            desc_str = search_description[key]
            term_freq[key] = desc_str.count(str(str1).lower())


def computeidf():
    if len(term_freq):
        return 1 + math.log2(len(description) / len(term_freq))
    else:
        return 0


def computetfidf(tf, idf):
    return tf * idf


def readfile():
    global readcsv,t,p
    global description,vocab_count,vocabulary
    stop_words = set(stopwords.words('english'))
    with open('data.csv', encoding="utf8") as csvfile:
        readcsv = csv.reader(csvfile, delimiter=',')
        for row in readcsv:
            description[row[32]] = row[6]
            key_names[row[32]] = row[17]
            word_tokens = nltk.word_tokenize(row[6])
            filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words]
            search_description[row[32]] = ''.join(filtered_sentence)
            class_rec[row[32]] = row[8]
            class_dict[row[8]].append(row[32])
            if row[8] == "1 Star hotel":
                for w in row[6]:
                    if w in stop_words:
                        pass
                    else:
                        if w in class1words:
                            class1words[str(w)] = 1
                        else:
                            class1words[str(w)] = 0
            if row[8] == "2 Star hotel":
                for w in word_tokens:
                    if w in stop_words:
                        pass
                    else:
                        if w in class2words:
                            class2words[w] += 1
                        else:
                            class2words[w] = 0
            if row[8] == "3 Star hotel":
                for w in word_tokens:
                    if not w in stop_words:
                        if w in class3words:
                            class3words[w] += 1
                        else:
                            class3words[w] = 0
            if row[8] == "4 Star hotel":
                for w in word_tokens:
                    if not w in stop_words:
                        if w in class4words:
                            class4words[w] += 1
                        else:
                            class4words[w] = 0
            if row[8] == "5 Star hotel":
                for w in word_tokens:
                    if not w in stop_words:
                        if w in class5words:
                            class5words[str(w)] += 1
                        else:
                            class5words[str(w)] = 0
            word_position = 0
            for w in word_tokens:
                # t.clear()
                # p.clear()
                if w.lower() in special_characters:
                    continue
                if w.lower() in vocabulary:
                    # t = vocabulary[str(w.lower())].values()
                    # p = t[str(row[32])]
                    # p[word_position]=word_position
                    # t[str(row[32])] = p
                    # vocabulary[str(w.lower())] = t
                    vocabulary[str(w.lower())] = vocabulary[str(w.lower())]+','+str(row[32]) + ',' + str(word_position)
                else:
                    vocabulary[str(w.lower())] = str(row[32])+','+str(word_position)
                word_position += 1
        for i in class_dict:
            class_probability[i] = len(class_dict[i])/5000
        j=1
        # total_count_words[j]=0
        # for i in class5words:
        #     total_count_words[j] += class5words[i]
        # j += 1
        # total_count_words[j] = 0
        # for i in class4words:
        #     total_count_words[j] += class4words[i]
        # j += 1
        # total_count_words[j] = 0
        # for i in class3words:
        #     total_count_words[j] += class3words[i]
        # j += 1
        # total_count_words[j] = 0
        # for i in class2words:
        #     total_count_words[j] += class2words[i]
        # j += 1
        # total_count_words[j] = 0
        # for i in class1words:
        #     total_count_words[j] += class1words[i]
        # j += 1
        # total_count_words[j] = 0
        # for i in class_dict:
        #     den_val[i] = total_count_words[j] + vocab_count


if __name__ == "__main__":
    app.run()
