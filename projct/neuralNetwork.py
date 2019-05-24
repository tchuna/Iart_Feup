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
from sklearn.svm import SVC
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

dataAll='data/allDataprocessed.csv.gz'
dataTest='data/testDataprocessed.csv.gz'
dataTrain='data/trainDataprocessed.csv.gz'




def neuralNetworkModel():


    allData=pd.read_csv(dataAll)
    print(allData.shape)


    allData['review'] =allData['review'].str.replace("&#039;", "'")
    allData['ReviewWithoutStopwords'] =allData['ReviewWithoutStopwords'].str.replace("&#039;", "'")


    tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,2))
    features = tfidf.fit_transform(allData.ReviewWithoutStopwords)
    labels   = allData.ratingSentimentLabel


    X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.20,random_state=0)
    normalize = Normalizer()

    X_train = normalize.fit_transform(X_train)
    X_test = normalize.transform(X_test)


    model1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 5), random_state=1,activation='tanh')

    print("\nStart to train  NN Models ")

    start = time.time()
    model1.fit(X_train,y_train)
    end=time.time()

    print("\nFinish to train  the model  hidden_layer_sizes=(8, 5)")
    print("Time to train the Model : ",int((end-start)),"seconds\n")


    result=[model1,X_test,y_test]

    print("Finish(Train)")


    return result


def  testModel(trainResult):

    model=trainResult[0]
    X_test=trainResult[1]
    y_test=trainResult[2]


    print("\nStart to test NN Models")

    start = time.time()
    y_test_pred = model.predict(X_test)
    end=time.time()

    print("\n Neural Network hidden_layer_sizes=(8, 5)\n")
    print("Time to test the Model : ",(end-start),"\n")
    print("Accuracy : ",accuracy_score(y_test, y_test_pred)*100,"%\n")
    print("Loss : ",hamming_loss(y_test, y_test_pred)*100,"%\n")
    print("\n",classification_report(y_test,y_test_pred,target_names=['neutre(5-7)','negative(0-3)','positive(8-10)']),"\n")
    print("Finish test \n")



    print("confusion_matrix NN hidden_layer_sizes=(8, 5)")
    conf_mat = confusion_matrix(y_test,y_test_pred)
    fig,ax = plot_confusion_matrix(conf_mat=conf_mat,colorbar=True,show_absolute=True,cmap='viridis')


    plt.show()


    return



def main():
    aux=neuralNetworkModel()
    testModel(aux)


main()
