"""
The process of feature scaling (also known as data normalization) is used to
standardize the range of data features. It becomes a necessary step in data
preprocessing when utilizing machine learning algorithms because the range of
data values can change greatly.
2. min-max normalization= (original_feature(x) - average(x) / Max(x) -min(x)
"""

################################ min-max normalization ##############################
import pandas as pd ## data processing, CSV file I/O (e.g. pd.read_csv)
df_train = pd.read_csv('D:/Houses-Regression/train.csv')
train=df_train
train.dtypes

########## get numeric data
train = train.select_dtypes(exclude=['object'])
train.head()
#########

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train = scaler.fit_transform(train)

#########

train=pd.DataFrame(train)
train.head()