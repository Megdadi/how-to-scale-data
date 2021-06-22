# -*- coding: utf-8 -*-
"""
The process of feature scaling (also known as data normalization) is used to
standardize the range of data features. It becomes a necessary step in data
preprocessing when utilizing machine learning algorithms because the range of
data values can change greatly.
3. Normlalization: work with rows
    A. The L1 norm that is calculated as the sum of the absolute values of the vector.
    B. The L2 norm that is calculated as the square root of the sum of the squared vector values.
    C. The max norm that is calculated as the maximum vector values.
    
"""
import pandas as pd ## data processing, CSV file I/O (e.g. pd.read_csv)
df_train = pd.read_csv('D:/Houses-Regression/train.csv')
train=df_train
train.dtypes

########## get numeric data
train = train.select_dtypes(exclude=['object'])
train.head()
######### Normalizer does not work with missing values
train.shape # (1460, 38)
train_missing=train.isnull().sum().sort_values(ascending= False)
train= train.drop(['LotFrontage', 'GarageYrBlt', 'MasVnrArea'], axis =1)
train.shape # (1460, 35)
from sklearn.preprocessing import Normalizer 
scaler = Normalizer(norm='l2') # you can change the norm to 'l1' or 'max' 
#transformer = Normalizer(norm='l2' )
#transformer = Normalizer(norm='max' )
train = scaler.fit_transform(train)

train=pd.DataFrame(train)
train.head()
######################################