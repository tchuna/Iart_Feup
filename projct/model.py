import sys
import numpy as np
import pandas as pd
import tensorflow as tf

import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import nltk
nltk.download(['punkt','stopwords'])

from tensorflow import keras
from datetime import datetime
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from nltk.corpus import stopwords




analyzer = SentimentIntensityAnalyzer()
stopwords = stopwords.words('english')


import matplotlib.pyplot as plt
import xlrd as xl

plt.style.use('ggplot')



dataTest='drugsComTest_raw.tsv'
dataTrain='drugsComTrain_raw.tsv'

trainData_index=0
testData_index=1




def read_Data():

    data_to_test= pd.read_csv(dataTest, delimiter='\t')
    data_to_train= pd.read_csv(dataTrain, delimiter='\t')



    data_to_test.columns = ['Id','drugName','condition','review','rating','date','usefulCount']
    data_to_train.columns= ['Id','drugName','condition','review','rating','date','usefulCount']

    data_to_test['date'] = pd.to_datetime(data_to_test['date'])
    data_to_train['date'] = pd.to_datetime(data_to_train['date'])


    result=[data_to_train,data_to_test]

    return result





def clean_Data(data):

    numCorrupD=len(data[trainData_index][data[trainData_index].condition.str.contains(" users found this comment helpful.",na=False)])
    print("Corrupted Reviews in Train Data:",numCorrupD,"\n")
    clean_train_data=data[trainData_index][~data[trainData_index].condition.str.contains(" users found this comment helpful.",na=False)]
    clean_train_data=shuffle(clean_train_data);

    clean_train_data = clean_train_data[['Id','review','rating']].copy()


    numCorrupD=len(data[testData_index][data[testData_index].condition.str.contains(" users found this comment helpful.",na=False)])
    print("Corrupted Reviews in Test Data:",numCorrupD,"\n")
    clean_test_data=data[testData_index][~data[testData_index].condition.str.contains(" users found this comment helpful.",na=False)]
    clean_test_data=shuffle(clean_test_data)

    clean_test_data = clean_test_data[['Id','review','rating']].copy()


    result=[clean_train_data,clean_test_data]

    return result




def processData(data):

    allData=clean_Data(data)

    allData[testData_index]['review'] = allData[testData_index]['review'].replace("&#039;", "'",regex=False)
    allData[trainData_index]['review'] = allData[trainData_index]['review'].replace("&#039;", "'",regex=False)

    #clean review //ReviewWithoutStopwords
    allData[trainData_index]['ReviewWithoutStopwords'] = allData[trainData_index]['review'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))     # remove stopwords from review
    allData[testData_index]['ReviewWithoutStopwords'] = allData[testData_index]['review'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))     # remove stopwords from review

    #score review whit vanderSentiment
    allData[trainData_index]['reviewScore'] = allData[trainData_index]['ReviewWithoutStopwords'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    allData[testData_index]['reviewScore'] = allData[testData_index]['ReviewWithoutStopwords'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    #map reviewScore in type(positive(2),neutre(0),negative(1))
    allData[trainData_index]['vaderReviweScore']= allData[trainData_index]['reviewScore'].map(lambda x:int(2) if x>=0.05 else int(1) if x<=-0.05 else int(0) )
    allData[testData_index]['vaderReviweScore']= allData[testData_index]['reviewScore'].map(lambda x:int(2) if x>=0.05 else int(1) if x<=-0.05 else int(0) )

    allData[trainData_index]['vaderReviweScoreLabel']= allData[trainData_index]['reviewScore'].map(lambda x:"positive" if x>=0.05 else "negative" if x<=-0.05 else "neutre" )
    allData[testData_index]['vaderReviweScoreLabel']= allData[testData_index]['reviewScore'].map(lambda x:"positive" if x>=0.05 else "negative" if x<=-0.05 else "neutre" )

    #rating Sentiment score
    allData[trainData_index]['ratingSentiment']= allData[trainData_index]['rating'].map(lambda x:int(2) if x>=7 else int(1) if x<=3 else int(0) )
    allData[testData_index]['ratingSentiment']= allData[testData_index]['rating'].map(lambda x:int(2) if x>=7 else int(1) if x<=3 else int(0) )

    allData[trainData_index]['ratingSentimentLabel']= allData[trainData_index]['rating'].map(lambda x:"positive" if x>=7 else "negative" if x<=3 else "neutre" )
    allData[testData_index]['ratingSentimentLabel']= allData[testData_index]['rating'].map(lambda x:"positive" if x>=7 else "negative" if x<=3 else "neutre" )




    print("number labelsReview train data\n",allData[trainData_index]['vaderReviweScoreLabel'].value_counts())
    print("number labelsReview test data\n",allData[testData_index]['vaderReviweScoreLabel'].value_counts())

    print("number labelsRating train data\n",allData[trainData_index]['ratingSentimentLabel'].value_counts())
    print("number labelsRating test data\n",allData[testData_index]['ratingSentimentLabel'].value_counts())




    return allData







def main():
    allData=read_Data();
    allcleanData= processData(allData)

    print(allcleanData[trainData_index].head(1))





main()
