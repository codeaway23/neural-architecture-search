import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

import pandas as pd

data = pd.read_csv('customer-churn.csv')
cat_cols=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService','OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data[cat_cols] = data[cat_cols].apply(lambda col:le.fit_transform(col))
data.drop(['customerID'], axis=1, inplace=True)
data.set_value(488,'TotalCharges', np.nan)
data.set_value(753,'TotalCharges', np.nan)
data.set_value(936,'TotalCharges', np.nan)
data.set_value(1082,'TotalCharges', np.nan)
data.set_value(1340,'TotalCharges', np.nan)
data.set_value(3331,'TotalCharges', np.nan)
data.set_value(3826,'TotalCharges', np.nan)
data.set_value(4380,'TotalCharges', np.nan)
data.set_value(5218,'TotalCharges', np.nan)
data.set_value(6670,'TotalCharges', np.nan)
data.set_value(6754,'TotalCharges', np.nan)
data['TotalCharges'] = data['TotalCharges'].astype(float)
data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
x = data[cat_cols].values[:,:-1]
y = data[cat_cols].values[:,-1]

target_classes = 2 ## binary classification

# data = pd.read_csv('fordTrain.csv')
# x = data.values[:,:-1]
# y = data.values[:,-1]
model = Sequential()
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x,y,epochs=200, validation_split=0.1)

# model = Sequential()
# model.add(Dense(32, activation='sigmoid'))
# model.add(Dense(32, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.fit(x,y,epochs=100, validation_split=0.1)
#
# model = Sequential()
# model.add(Dense(32, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.fit(x,y,epochs=100, validation_split=0.1)
#
# model = Sequential()
# model.add(Dense(16, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.fit(x,y,epochs=100, validation_split=0.1)
#
# model = Sequential()
# model.add(Dense(32, activation='sigmoid'))
# model.add(Dense(16, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.fit(x,y,epochs=100, validation_split=0.1)
#
# model = Sequential()
# model.add(Dense(128, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.fit(x,y,epochs=100, validation_split=0.1)
#
# model = Sequential()
# model.add(Dense(128, activation='sigmoid'))
# model.add(Dense(64, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.fit(x,y,epochs=100, validation_split=0.1)
#
# model = Sequential()
# model.add(Dense(64, activation='sigmoid'))
# model.add(Dense(16, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.fit(x,y,epochs=100, validation_split=0.1)
