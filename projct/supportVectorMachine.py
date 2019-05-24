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
from sklearn.metrics import roc_curve


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
tarin='data/trainDataprocessed.csv.gz'




def svnModel():


    allData=pd.read_csv(all)
    #print(allData.shape)


    allData['review'] =allData['review'].str.replace("&#039;", "'")
    allData['ReviewWithoutStopwords'] =allData['ReviewWithoutStopwords'].str.replace("&#039;", "'")

    allData=allData[['Id','reviewScore','vaderReviweScore','vaderReviweScoreLabel','ReviewWithoutStopwords','ratingSentiment']].copy()



    tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,2))


    features = tfidf.fit_transform(allData.ReviewWithoutStopwords)

    labels= allData.ratingSentiment



    print(features.shape)



    model = LinearSVC('l2',tol=1e-5)


    X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.10,random_state=0)

    normalize = Normalizer()

    X_train = normalize.fit_transform(X_train)
    X_test = normalize.transform(X_test)


    start = time.time()
    history=model.fit(X_train,y_train)
    end=time.time()


    print("Finish to train  the model")
    print("\nTime to train the Model : ",(end-start),"seconds")


    result=[model,X_test,y_test]

    return result



def  testModel(trainResult):

    model=trainResult[0]
    X_test=trainResult[1]
    y_test=trainResult[2]

    print("\nStart to test The Model")

    start = time.time()
    y_test_pred = model.predict(X_test)
    end=time.time()

    print("\nTime to test the Model : ",(end-start),"seconds")

    print("\nSupport Vector Machine Accuracy : ",accuracy_score(y_test, y_test_pred)*100,"%")
    print("\nSupport Vector Machine Loss : ",hamming_loss(y_test, y_test_pred)*100,"%")

    conf_mat = confusion_matrix(y_test,y_test_pred)
    fig,ax = plot_confusion_matrix(conf_mat=conf_mat,colorbar=True,show_absolute=True,cmap='viridis')

    print(classification_report(y_test,y_test_pred,target_names=['neutre(5-7)','negative(0-3)','positive(8-10)']))
    plt.show()


    return











def main():
    aux=svnModel()
    testModel(aux)


main()
