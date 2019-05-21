import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix




from tensorflow import keras
from datetime import datetime
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import xlrd as xl

plt.style.use('ggplot')

dataAll='allDataprocessed.csv'





def sentimentalModels():

    allData=pd.read_csv(dataAll)


    allData['review'] =allData['review'].str.replace("&#039;", "'")
    allData['ReviewWithoutStopwords'] =allData['ReviewWithoutStopwords'].str.replace("&#039;", "'")


    allData = allData[['Id','drugName','review','ReviewWithoutStopwords','rating','ratingSentiment','ratingSentimentLabel',
              'reviewScore','vaderReviweScore','vaderReviweScoreLabel']]



    tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,2))
    features = tfidf.fit_transform(allData.ReviewWithoutStopwords)
    labels   = allData.vaderReviweScore

    print(features.shape)



    """models = [RandomForestClassifier(n_estimators=200,max_depth=3,random_state=0),
              LinearSVC(),
              MultinomialNB(),
              LogisticRegression(random_state=0,solver='lbfgs',max_iter=2000,multi_class='auto')]"""




    x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.25,random_state=0)

    """model = LinearSVC('l2')
    x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.25,random_state=0)
    normalize = Normalizer()


    x_train = normalize.fit_transform(x_train)
    x_test = normalize.transform(x_test)

    model.fit(x_train,y_train)

    y_test_pred = model.predict(x_test)
    print(accuracy_score(y_test, y_test_pred))"""



    scaler = StandardScaler(with_mean=False)
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)


    #mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=10)
    mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=5)
    mlp.fit(x_train, y_train)

    predictions = mlp.predict(x_test)
    print(accuracy_score(y_test,predictions))




    # Plotting decision regions

    """plot_decision_regions(np.asarray(x_train)[0], np.asarray(y_train)[0], clf=model, legend=2)

    # Adding axes annotations
    plt.xlabel('Review')
    plt.ylabel('Review Type')
    plt.title('SVM on Drug Analize')
    plt.show()





    #print(model.accuracy.mean())


    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []

    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model,features,labels,scoring='accuracy',cv=CV)
        for fold_idx,accuracy in enumerate(accuracies):
            entries.append((model_name,fold_idx,accuracy))

    cv_df = pd.DataFrame(entries,columns=['model_name','fold_idx','accuracy'])

    print(cv_df)

    print(cv_df.groupby('model_name').accuracy.mean())"""






    return allData




def main():

    data=sentimentalModels()



main()
