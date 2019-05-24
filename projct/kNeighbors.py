import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import hamming_loss

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
from  sklearn.metrics import classification_report




from tensorflow import keras
from datetime import datetime
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import xlrd as xl

plt.style.use('ggplot')

all='data/allDataprocessed.csv.gz'
test='data/testDataprocessed.csv.gz'
train='data/trainDataprocessed.csv.gz'



def kNeighborsModel():
    allData=pd.read_csv(all)
    print(allData.shape)


    allData['review'] =allData['review'].str.replace("&#039;", "'")
    allData['ReviewWithoutStopwords'] =allData['ReviewWithoutStopwords'].str.replace("&#039;", "'")



    tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,2))
    features = tfidf.fit_transform(allData.ReviewWithoutStopwords)
    labels   = allData.vaderReviweScore

    X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.20,random_state=0)
    normalize = Normalizer()


    X_train = normalize.fit_transform(X_train)
    X_test = normalize.transform(X_test)

    model=KNeighborsClassifier(n_neighbors=1)
    #model=KNeighborsClassifier(n_neighbors=3)
    #model=KNeighborsClassifier(n_neighbors=5)

    print("\nStart to train  KNN Model ")

    start = time.time()
    model.fit(X_train,y_train)
    end=time.time()

    print("\nFinish to train  the model K=5")
    print("Time to train the Model : ",(end-start),"\n")

    result=[model,X_test,y_test]

    print("Finish(Train)")


    return result





def  testModel(trainResult):

    model=trainResult[0]
    X_test=trainResult[1]
    y_test=trainResult[2]


    print("\nStart to test KNN Models")

    start = time.time()
    y_test_pred = model.predict(X_test)
    end=time.time()

    print("\nSupport kNeighbors K=1\n")
    print("Time to test the Model : ",int((end-start)),"seconds\n")
    print("Support kNeighbors Accuracy : ",accuracy_score(y_test, y_test_pred)*100,"%\n")
    print("Support  kNeighbors Loss : ",hamming_loss(y_test, y_test_pred)*100,"%\n")
    print("\n",classification_report(y_test,y_test_pred,target_names=['neutre(5-7)','negative(0-3)','positive(8-10)']),"\n")
    print("Finish test K=1\n")

    print("\nFinish(Test)")

    print("confusion_matrix K=1")
    conf_mat = confusion_matrix(y_test,y_test_pred)
    fig,ax = plot_confusion_matrix(conf_mat=conf_mat,colorbar=True,show_absolute=True,cmap='viridis')


    plt.show()



    return





def main():
    aux=kNeighborsModel()
    testModel(aux)

main()
