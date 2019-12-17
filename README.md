# DataMiningProject
Data Mining Project

Hotel Search

Overview
It is basically an web application to search and rank the hotels based on the keywords that is entered.

Script in detail
Index.html
This is the main search page that is used for searching the data and to display the results. This HTML page displays the results in paragraph tag and contains a function to clear the previous search results.

```
<head>
		<meta charset="utf-8">
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<meta http-equiv="X-UA-Compatible" content="ie=edge">
		<link rel="stylesheet" href="{{ url_for('static', filename='main.css') }}">
		<title>Homepage</title>
		<script>
				function clear() {
						var el = document.getElementById(id);
						if (el) {
								el.innerHTML = '';
		}
}
		</script>
</head>
<body>
<h1>Search Hotels</h1>
	<form method="POST" action="/search">
		text : <input type="text" name="Search text" /><br /><br />
		<input type="submit" name="submit" value="search">
				<input type="submit" name="clear" value="clear">
	</form>
		{% for key, value in result.items() %}
				<h2> {{ key }} </h2>
				<p> {{ tfidf_display[key] }} </p>
				<p><blockquote> {{ value }} </blockquote>
		{% endfor %}

</body>
</html>
```


flask_app.py
This is the main python file where the program is running.

First we download the required files into the working directory
```
from flask import Flask, render_template, url_for, request
import csv
from nltk.stem import WordNetLemmatizer
import math
import time
import nltk
from nltk.corpus import stopwords
import jinja2
import threading

start_time = time.time()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

Read the dataset and store the data in the dictionary. Also we do preprocessing and removing the stopwords.
```
def readfile():
    global readcsv
    global description
    stop_words = set(stopwords.words('english'))
    with open('data.csv', encoding="utf8") as csvfile:
        readcsv = csv.reader(csvfile, delimiter=',')
        for row in readcsv:
            description[row[32]] = row[6]
            key_names[row[32]] = row[17]
            word_tokens = nltk.word_tokenize(row[6])
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            # description[row[32]] = ''.join(filtered_sentence)
            search_description[row[32]] = ''.join(filtered_sentence)
```

Compute tf and normalise it
```
def computetf(str):
    # for key,value in description.items():
    #     if str in value:
    #         desc_str = description[key]
    #         term_freq[key] = desc_str.count(str)
    for key,value in search_description.items():
        if str in value:
            desc_str = search_description[key]
            term_freq[key] = desc_str.count(str)/len(desc_str)
```

Compute idf with the formula of 1 plus log(Total number of ducments divided by number of documents containing the search word) to the base 2
```
def computetf(str):
    # for key,value in description.items():
    #     if str in value:
    #         desc_str = description[key]
    #         term_freq[key] = desc_str.count(str)
    for key,value in search_description.items():
        if str in value:
            desc_str = search_description[key]
            term_freq[key] = desc_str.count(str)/len(desc_str)
```

Computing tf-idf
```
def computeidf():
    if len(term_freq):
        return 1 + math.log2(len(description)/len(term_freq))
    else:
        return 0
```

On the opening of the webpage
```
@app.route('/')
def index():
    readfile()
    return render_template('index.html', result=search_result, tfidf_display=tfidf_display, text=text)

We are using lemmetization so that the search results are better
```
                    lemmatizer = WordNetLemmatizer()
                    string[i] = lemmatizer.lemmatize(string[i])
                    found = False
                    for key, value in description.items():
                        found = True
                        if string[i] in value:
                            computetf(string[i])
                            idf = computeidf()
                            for key1, value1 in term_freq.items():
                                tf_idf[key1] = computetfidf(term_freq[key1], idf)
																```
																
																
On the click of search we are using the POST method and calculating the tf-idf and ranking the results. We use render template to 
display the results from python on the HTML page 

```
@app.route("/search", methods=['POST', 'GET'])
def search():
    if request.method == 'POST':
        text = request.form['Search text']
        tf_idf_full.clear()
        tfidf_display.clear()
        if text:
            string = text.split(" ")
            found = False

            for i in range(len(string)):
                tf_idf.clear()

                for key, value in description.items():
                    if string[i] in value:
                        found = True
                        # found = False
                        computetf(string[i])
                        idf = computeidf()
                        for key1,value1 in term_freq.items():
                            tf_idf[key1] = computetfidf(term_freq[key1],idf)
                if not found:
                    lemmatizer = WordNetLemmatizer()
                    string[i] = lemmatizer.lemmatize(string[i])
                    found = False
                    for key, value in description.items():
                        found = True
                        if string[i] in value:
                            computetf(string[i])
                            idf = computeidf()
                            for key1, value1 in term_freq.items():
                                tf_idf[key1] = computetfidf(term_freq[key1], idf)
                for key2,value2 in tf_idf.items():
                    if key2 in tf_idf_full:
                        tf_idf_full[key2] = tf_idf_full[key2] + tf_idf[key2]
                    else:
                        tf_idf_full[key2] = tf_idf[key2]
                    tfidf_display[key_names[key2]] = tf_idf_full[key2]
            display(text)
            print("--- %s seconds ---" % (time.time() - start_time))
            return render_template('index.html', result=search_result, tfidf_display=tfidf_display, text=text)
        else:
            term_freq.clear()
            tf_idf.clear()
            tf_idf_full.clear()
            tfidf_display.clear()
            search_result.clear()
            return render_template('index.html', result=search_result, tfidf_display=tfidf_display, text=text)
    else:
        return render_template('index.html')
```


Hotel Classifier
Here i have used Naive bayes theorem for classification. So in this we would need to calculate the conditional probabilities given the hypothesis.
After calculating the accuracies i have found that Naive bayes works the best for my dataset so have implemented Naive bayes classifier from sratch.

```
def classify():
    global class1count,class2count,class3count,class4count,class5count
    vid = pandas.read_csv("data.csv")
    x_train, x_test, y_train, y_test= train_test_split(vid['hotel_description'], vid['hotel_star_rating'], test_size=0.20, random_state=100, shuffle=False)
    count_vector = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',lowercase=True, stop_words='english')
    x_train_cv = count_vector.fit_transform(x_train.values.astype('U'))
    x_test_cv = count_vector.transform(x_test.values.astype('U'))
    text = request.form['Classify text']
    string = text.split(" ")
    p_c1=1
    p_c2=1
    p_c3=1
    p_c4=1
    p_c5=1
    getcontext().prec = 5
    present = False
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
                present =True
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
        if (class1count == 1) and (class2count == 1) and (class3count == 1) and (class4count == 1) and (class5count == 1):
            p_word_class1 = 1
            p_word_class2 = 1
            p_word_class3 = 1
            p_word_class4 = 1
            p_word_class5 = 1
        else:
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
    if present == False:
        for s in range(len(text.split(" "))):
            print(string[s])
            for syn in wordnet.synsets(string[s]):
                for l in syn.lemmas():
                    if str(l.name()) in string:
                        pass
                    else:
                        string.append(str(l.name()))
        print(string)
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
                    present = True
                    term_documents[key] = 1
                    x = class_rec[key]
                    if x == "1 Star hotel":
                        class1count += 1
                    elif x == "2 Star hotel":
                        class2count += 1
                    elif x == "3 Star hotel":
                        class3count += 1
                    elif x == "4 Star hotel":
                        class4count += 1
                    elif x == "5 Star hotel":
                        class5count += 1
            if (class1count == 1) and (class2count == 1) and (class3count == 1) and (class4count == 1) and (
                    class5count == 1):
                p_word_class1 = 1
                p_word_class2 = 1
                p_word_class3 = 1
                p_word_class4 = 1
                p_word_class5 = 1
            else:
                p_word_class1 = class1count / len(class_dict["1 Star hotel"])
                p_word_class2 = class2count / len(class_dict["2 Star hotel"])
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
    text = str(string).replace("'", "")
    text = str(text).replace("[", "")
    text = str(text).replace("]", "")
    return render_template('classifier.html', result=search_result_classify, text=text) 
  

For calculating the tf idf for the image recognition system i have used the phase 1 code lines
```
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
	   
	   
For classification experiments the following code lines have been used
We need to calculate the accuracy of various models
Have used libraries for calculating the accuraies of various models and used prediction methods.

```
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
```

For the Phrase search:
Algorithm code lines:
```
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
                text=str(text).replace("\"","")
