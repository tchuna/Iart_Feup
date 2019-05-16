import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from pandas import ExcelWriter
from pandas import ExcelFile
from tensorflow import keras
from datetime import datetime
from sklearn.preprocessing import normalize, MinMaxScaler

import matplotlib.pyplot as plt
import xlrd as xl

plt.style.use('ggplot')
dataTest='drugsComTest_raw.tsv'
dataTrain='drugsComTrain_raw.tsv'

testData=0;
trainData=1
allData=2




def read_Data():

    df1 = pd.read_csv(dataTest, delimiter='\t')
    df2 = pd.read_csv(dataTrain, delimiter='\t')

    drugsData=pd.concat([df1,df2])
    drugsData.columns = ['ID','drugName','condition','review','rating','date','usefulCount']
    drugsData['date'] = pd.to_datetime(drugsData['date'])

    result=[df1,df2,drugsData]

    return result



def clean_Data(drugsData):
    numCorrupD=len(drugsData[drugsData.condition.str.contains(" users found this comment helpful.",na=False)])

    print("Corrupted Reviews:",numCorrupD,"\n")

    drugsDataCleaned=drugsData[~drugsData.condition.str.contains(" users found this comment helpful.",na=False)]

    return drugsDataCleaned




def descriptiveAnalisis(data):
    print("Reviews Data Describe:")
    print(data.describe(),"\n")

    # Number of reviews per drugs
    reviews_per_drug = data.groupby(["drugName"]).agg({
    "ID": pd.Series.nunique
    })

    print("Number of Reviews per Drug")
    print(reviews_per_drug.describe(),"\n")

    # Number of reviews per condition
    reviews_per_condition = data.groupby(["condition"]).agg({
        "ID": pd.Series.nunique
    })

    print("Number of Reviews per Condition")
    print(reviews_per_condition.describe(),"\n")

    # Number of reviews per date
    reviews_per_date = data.groupby(["date"]).agg({
        "ID": pd.Series.nunique
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
    'ID': pd.Series.nunique
    })

    #Significant review number
    significantRevNumb=int(reviews_per_drug.quantile(q=0.75).values[0])
    drugs_rating = drugs_rating[drugs_rating['ID'] > significantRevNumb]

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
    'ID': pd.Series.nunique
    })

    print("Number of pairs (Drug, Condition):", len(drugs_condition_rating),"\n")

    significantRevNumb=int(drugs_condition_rating['ID'].quantile(q=0.75))
    print("Significant number of reviews (Drug, Condition): More than",significantRevNumb, "reviews","\n")

    drugs_condition_rating = drugs_condition_rating[drugs_condition_rating['ID'] >significantRevNumb]

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
        'ID': pd.Series.nunique
        })

    different_months = len(days_grouped)

    print("Months on dataset(In Different Years): ", different_months,"\n")

    MME = MinMaxScaler() # Min-max normalization (0-1) for better visualization

    grouped = days_grouped.reset_index(level=1)
    index_values = np.unique(grouped.index.values)[1:] # First year (2008) month of January is missing

    months = pd.DataFrame()

    for year in index_values:
        months[year] = grouped.loc[year,:]["rating"].values # Every column is a year


    months_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dic"]

    months.iloc[:,:] = MME.fit_transform(months) # Min Max Normalization by columns (year)

    plots = months.plot(subplots=True, legend=True, figsize=(8,18), lw=2, title="Normalized (min-max) ratings average given in reviews for every month in every year")

    for plot in plots:
        x = plot.set_ylim([-0.05, 1.05]) # Just assigning to variable so there is no output on notebook


    x = plt.xticks(range(0, len(months_labels)), months_labels)


    plt.show()






def main():
    data=read_Data()


    dataCleaned=clean_Data(data[allData])

    descriptiveAnalisis(dataCleaned)

main()
