import numpy as np
import pandas as pd
import tensorflow as tf

from pandas import ExcelWriter
from pandas import ExcelFile
from tensorflow import keras
import xlrd as xl




def read_Data(file_name):
    data = pd.read_excel(file_name, sheet_name='drugsComTest_raw')
    return data



def main():
    data=read_Data('drugsComTest_raw.xlsx')

    drugName =data['drugName'].unique()
    dataUpper= [x.upper() for x in drugName]
    dataUpper.sort()

    for x in dataUpper:
        print(x)
    #print(data['drugName'].unique())



main()
