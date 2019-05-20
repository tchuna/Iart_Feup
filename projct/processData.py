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


import matplotlib.pyplot as plt
import xlrd as xl




analyzer = SentimentIntensityAnalyzer()
stopwords = stopwords.words('english')



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

    #clean_train_data = clean_train_data[['Id','review','rating']].copy()


    numCorrupD=len(data[testData_index][data[testData_index].condition.str.contains(" users found this comment helpful.",na=False)])
    print("Corrupted Reviews in Test Data:",numCorrupD,"\n")
    clean_test_data=data[testData_index][~data[testData_index].condition.str.contains(" users found this comment helpful.",na=False)]
    clean_test_data=shuffle(clean_test_data)

    #clean_test_data = clean_test_data[['Id','review','rating']].copy()


    result=[clean_train_data,clean_test_data]

    return result




def processData(dataTo_Process):

    allData=clean_Data(dataTo_Process)
    data=pd.concat([allData[trainData_index],allData[testData_index]])
    data.column= ['Id','drugName','condition','review','rating','date','usefulCount']

    allData[testData_index]['review'] = allData[testData_index]['review'].str.replace("&#039;", "'")
    allData[trainData_index]['review'] = allData[trainData_index]['review'].str.replace("&#039;", "'")
    data['review']=data['review'].str.replace("&#039;", "'")

    print("Reviews Data Describe:")
    print(data.describe(),"\n")

    # Number of reviews per drugs
    reviews_per_drug = data.groupby(["drugName"]).agg({
    "Id": pd.Series.nunique
    })

    print("Number of Reviews per Drug")
    print(reviews_per_drug.describe(),"\n")

    # Number of reviews per condition
    reviews_per_condition = data.groupby(["condition"]).agg({
        "Id": pd.Series.nunique
    })

    print("Number of Reviews per Condition")
    print(reviews_per_condition.describe(),"\n")

    # Number of reviews per date
    reviews_per_date = data.groupby(["date"]).agg({
        "Id": pd.Series.nunique
    })

    print("Number of Reviews per date")
    print(reviews_per_date.describe(),"\n")


    #top 10 # reviewes drugs
    plt.figure(1)
    plot = data.drugName.value_counts().nlargest(10).plot(kind='bar',
                                                   title="Top 10 reviewed drugs", figsize=(6,3))
    plt.figure(2)

    # Top 10 most suffered condition by reviewers
    plot = data.condition.value_counts().nlargest(10).plot(kind='bar',
                                                     title="Top 10 conditions in reviews", figsize=(6,3))


    drugs_rating = data.groupby('drugName').agg({
    'rating': np.mean,
    'Id': pd.Series.nunique
    })

    #Significant review number
    significantRevNumb=int(reviews_per_drug.quantile(q=0.75).values[0])
    drugs_rating = drugs_rating[drugs_rating['Id'] > significantRevNumb]

    print("Significant number of reviews: More than",significantRevNumb, "reviews","\n")

    #top 20
    top_drugs_rating = drugs_rating.nlargest(20, 'rating')
    plot = top_drugs_rating.plot(y='rating', kind='bar', figsize = (6, 3))
    fig0 = plt.title("Top 20 'significant' drugs with best rating")
    fig0 = plt.ylim(9, 10) #ranking output [9-10]


    #bottom 20
    bottom_drugs_rating = drugs_rating.nsmallest(20, 'rating')
    plot = bottom_drugs_rating.plot(y='rating', kind='bar', figsize = (6, 3))
    fig1 = plt.title("Top 20 'significant' drugs with worst rating")
    fig1 = plt.ylim(1, 5) #ranking output [1-5]




    drugs_condition_rating = data.groupby(['drugName', 'condition']).agg({
    'rating': np.mean,
    'Id': pd.Series.nunique
    })

    print("Number of pairs (Drug, Condition):", len(drugs_condition_rating),"\n")

    significantRevNumb=int(drugs_condition_rating['Id'].quantile(q=0.75))
    print("Significant number of reviews (Drug, Condition): More than",significantRevNumb, "reviews","\n")

    drugs_condition_rating = drugs_condition_rating[drugs_condition_rating['Id'] >significantRevNumb]

    #top 20
    top_drugs_condition_rating = drugs_condition_rating.nlargest(20, 'rating')
    plot = top_drugs_condition_rating.plot(y='rating', kind='bar', figsize = (6, 3))
    fig2 = plt.title("Top 20 (Drug - Condition) with best rating")
    fig2 = plt.ylim(9, 10)

    #bottom 20
    bottom_drugs_condition_rating = drugs_condition_rating.nsmallest(20, 'rating')
    plot = bottom_drugs_condition_rating.plot(y='rating', kind='bar', figsize = (6, 3))
    dummy = plt.title("Top 10 (Drug - Condition) with worst rating")
    dummy = plt.ylim(1, 5)


    #temporal analize


    start_date = data["date"].min()
    end_date = data["date"].max()

    print("First review date: ", start_date)
    print("Last review date: ", end_date,"\n")

    data["month"] = data["date"].apply(lambda x: x.strftime('%m')) # Extract date month
    data["year"] = data["date"].apply(lambda x: x.strftime('%Y')) # Extract date year
    data["weekday"] = data["date"].apply(lambda x: x.strftime('%w')) # Extract date weekday


    days_grouped = data.groupby(["year", "month"])
    days_grouped = days_grouped.agg({
        'rating': np.mean,
        'usefulCount': np.sum,
        'Id': pd.Series.nunique
        })

    different_months = len(days_grouped)

    print("Months on dataset(In Different Years): ", different_months,"\n")

    MME = MinMaxScaler() # Min-max normalization (0-1) for better visualization

    grouped = days_grouped.reset_index(level=1)
    index_values = np.unique(grouped.index.values)[1:] # First year (2008) month of January is missing

    months = pd.DataFrame()

    for year in index_values:
        months[year] = grouped.loc[year,:]["rating"].values # Every column is a year


    months_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    months.iloc[:,:] = MME.fit_transform(months) # Min Max Normalization by columns (year)

    plots = months.plot(subplots=True, legend=True, figsize=(8,18), lw=2, title="Normalized (min-max) ratings average given in reviews for every month in every year")

    for plot in plots:
        x = plot.set_ylim([-0.05, 1.05]) # Just assigning to variable so there is no output on notebook


    x = plt.xticks(range(0, len(months_labels)), months_labels)


    plt.show()




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

    allData[trainData_index].to_csv('trainDataprocessed.csv')
    allData[testData_index].to_csv('testDataprocessed.csv')
    conctData=pd.concat([allData[trainData_index],allData[testData_index]])
    conctData.to_csv('AllDataprocessed.csv')

    trainData=allData[trainData_index]
    testData=allData[testData_index]


    trainData = trainData[['Id','drugName','review','ReviewWithoutStopwords','rating','ratingSentiment','ratingSentimentLabel',
                  'reviewScore','vaderReviweScore','vaderReviweScoreLabel']]

    testData = testData[['Id','drugName','review','ReviewWithoutStopwords','rating','ratingSentiment','ratingSentimentLabel',
                  'reviewScore','vaderReviweScore','vaderReviweScoreLabel']]




    print(trainData.groupby('vaderReviweScoreLabel').size())
    trainData.groupby('vaderReviweScoreLabel').count().plot.bar()
    plt.show()

    print(trainData.groupby('ratingSentimentLabel').size())
    trainData.groupby('ratingSentimentLabel').count().plot.bar()
    plt.show()

    positive_vader_sentiments = trainData[trainData.ratingSentiment == 2]
    positive_string = []

    for s in positive_vader_sentiments.ReviewWithoutStopwords:
        positive_string.append(s)

    positive_string = pd.Series(positive_string).str.cat(sep=' ')

    plt.suptitle('positive_vader_sentiments')
    wordcloud = WordCloud(width=2000,height=1000,max_font_size=200).generate(positive_string)
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.show()


    negative_vader_sentiments = trainData[trainData.ratingSentiment == 1]
    negative_string = []

    for s in negative_vader_sentiments.ReviewWithoutStopwords:
        negative_string.append(s)

    negative_string = pd.Series(negative_string).str.cat(sep=' ')

    plt.suptitle('negative_vader_sentiments')
    wordcloud = WordCloud(width=2000,height=1000,max_font_size=200).generate(negative_string)
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.show()


    neutral_vader_sentiments = trainData[trainData.ratingSentiment == 0]
    neutral_string = []

    for s in neutral_vader_sentiments.ReviewWithoutStopwords:
        neutral_string.append(s)

    neutral_string = pd.Series(neutral_string).str.cat(sep=' ')

    plt.suptitle('neutre_vader_sentiments')
    wordcloud = WordCloud(width=2000,height=1000,max_font_size=200).generate(neutral_string)
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.show()



    return allData




def main():
    allData=read_Data();
    allcleanData= processData(allData)

main()
