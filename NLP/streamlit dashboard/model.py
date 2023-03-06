import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib as jb

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42, stratify = y)


tfidf = TfidfVectorizer(max_features= 2500, min_df= 2)
X_train = tfidf.fit_transform(X_train).toarray()
X_test = tfidf.transform(X_test).toarray()



def train_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    precision = round(precision_score(y_test, y_pred), 3)
    recall = round(recall_score(y_test, y_pred), 3)

    print(f'Accuracy of the model: {accuracy}')
    print(f'Precision Score of the model: {precision}')
    print(f'Recall Score of the model: {recall}')

    sns.set_context('notebook', font_scale= 1.3)
    fig, ax = plt.subplots(1, 2, figsize = (25,  8))
    ax1 = plot_confusion_matrix(y_test, y_pred, ax= ax[0], cmap= 'YlGnBu')
    ax2 = plot_roc(y_test, y_prob, ax= ax[1], plot_macro= False, plot_micro= False, cmap= 'summer')




def plot_confusion_matrix(y_true, y_pred, ax=None, cmap='Blues'):
    labels = list(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ax = sns.heatmap(cm, annot=True, cmap=cmap, fmt='d', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return ax

def plot_roc(y_test, y_prob, ax, plot_macro=False, plot_micro=False, cmap='summer'):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    return ax

from sklearn.metrics import roc_curve, auc
nb = MultinomialNB()
train_model(nb)



filename = 'nlp_model.joblib'
jb.dump(nb,filename)
print("done")