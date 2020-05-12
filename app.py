#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.externals import joblib
from flask import Flask ,render_template ,request
import re

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

#load model
classifier = joblib.load('model/Ridge_model.sav')
tfidf = joblib.load('model/tfidf_model.sav')

#set up tfidfvectorizor
tfidf_vectorizor=TfidfVectorizer(stop_words='english', max_df=0.7)

app= Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        x=request.form['review']
        tfidf_review=tfidf.transform([x,])
        y_pred=classifier.predict(tfidf_review)
        result=str(np.round(y_pred)).strip("[.]")
        return render_template('index.html',result=result)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)