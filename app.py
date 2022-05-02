import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle
import pandas as pd
df = pd.read_csv('Data.csv')

app = Flask(__name__)
model = pickle.load(open('model_pkl', 'rb'))

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),max_df=0.75, min_df=5, max_features=10000)
tfidf = tfidf_vectorizer.fit_transform(df['Text'] )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    if request.method == 'GET':
        data=request.args.get('data')
        print(data)
        tfidf1= tfidf_vectorizer.transform([data])
        prediction = model.predict(tfidf1)
        output = int(prediction[0])
        if output == 0:
            text = "No bully"
        elif output == 1:
            text = "Foul language"
        elif output == 2:
            text = "Light bully"
        elif output == 3:
            text = "Medium bully"
        elif output == 4:
            text = "Severe bully"

    return render_template('test.html', prediction_text='Bully severity is {}'.format(text))


if __name__ == "__main__":
    app.run(debug=True)

