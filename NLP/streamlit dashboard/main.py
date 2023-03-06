from flask import Flask, render_template, request
from joblib import load
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)

# Load the model from the joblib file
model = load('nlp_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
# Define a route for the prediction form
@app.route('/', methods=['GET', 'POST'])
def predict():
    
    Mental_Health_Twitter_data=pd.read_csv('/Users/ds_learner__22/Downloads/Mental-Health-Twitter (2).csv')
    Mental_Health_Twitter_data.head()

    Mental_Health_Twitter_data=Mental_Health_Twitter_data.drop("post_id",axis=1)
    #Mental_Health_Twitter_data=Mental_Health_Twitter_data.drop("post_created",axis=1)
    Mental_Health_Twitter_data=Mental_Health_Twitter_data.drop("user_id",axis=1)
    Mental_Health_Twitter_data['post_created'] = pd.to_datetime(Mental_Health_Twitter_data['post_created'])
    Mental_Health_Twitter_data["Month"] = Mental_Health_Twitter_data.post_created.dt.month

    Mental_Health_Twitter_data["Year"]= Mental_Health_Twitter_data.post_created.dt.year
    Mental_Health_Twitter_data["day"]= Mental_Health_Twitter_data.post_created.dt.day

    Mental_Health_Twitter_data=Mental_Health_Twitter_data.drop("post_created",axis=1)

    X = Mental_Health_Twitter_data["post_text"]
    y = Mental_Health_Twitter_data['label'].values
    
    cv = CountVectorizer()
    X=cv.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42, stratify = y)


    

    
    # X_train = [str(item) for item in X_train]
    # X_test = [str(item) for item in X_test]
    
    # tfidf = TfidfVectorizer(max_features=5000000, min_df=2, lowercase=False)
    # X_train = tfidf.fit_transform(X_train).toarray()
    # X_test = tfidf.transform(X_test).toarray()
    
    
    
    
    nb = MultinomialNB()
    nb.fit(X_train,y_train)
    nb.score(X_test,y_test)

    prediction = None
   
    if request.method == 'POST':
        # Get the input data
        text = request.form['text']
        data = [text]
        vect = cv.transform(data).toarray()
        # Perform the prediction
        prediction = nb.predict(vect)
        
        if predict == 0:
            prediction="Negative"
        else:
            prediction="Positive"
    # Render the prediction template
    return render_template('results.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True,port=5001)

