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
```


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




