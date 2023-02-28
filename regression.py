#!/usr/bin/env python
# coding: utf-8

# In[ ]:


cd C:/Users/frankh/python/regression_challenge


# In[ ]:


pwd


# In[ ]:


import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import tensorflow        as tf
import os
import random
import time
import datetime

from matplotlib                 import pyplot as pl

from tensorflow	                import keras
from tensorflow.keras.models    import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras           import layers
from tensorflow.keras.layers    import InputLayer
from tensorflow.keras.layers    import Dense
from tensorflow.keras.layers    import Dropout

from sklearn.model_selection     import train_test_split
from sklearn.preprocessing       import StandardScaler
from sklearn.preprocessing       import Normalizer
from sklearn.metrics             import r2_score
from sklearn.compose             import ColumnTransformer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import EarlyStopping

debug = False

def reset_seeds():
    seed_value = 1234
    random.seed(seed_value)
    np.random.seed(seed_value) 
    tf.random.set_seed(seed_value)
    tf.compat.v1.set_random_seed(seed_value)
    tf.random.set_seed(seed_value)


# In[ ]:


unix1 = datetime.datetime.timestamp(datetime.datetime.now())*1000
print(unix1)
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())
reset_seeds()
unix2 = datetime.datetime.timestamp(datetime.datetime.now())*1000
print(unix2)


# In[ ]:


dataset = pd.read_csv("admissions_data.csv")
print(dataset.describe(include='all'))
print(dataset.head())
print(dataset.columns)


# In[ ]:


features = dataset.iloc[:,1:-1]
labels   = dataset.iloc[:,-1]
print(features.columns)


# In[ ]:


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=1)


# In[ ]:


def print_column_values(dataset):
    if debug :
        print(dataset['GRE Score'].unique())
        print(dataset['TOEFL Score'].unique())
        print(dataset['University Rating'].unique())
        print(dataset['SOP'].unique())
        print(dataset['LOR '].unique())
        print(dataset['CGPA'].unique())
        print(dataset['Research'].unique())


# In[ ]:


debug = False
print_column_values(features)


# In[ ]:


numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns
print(numerical_columns)


# In[ ]:


ct = ColumnTransformer([('scale',StandardScaler(),  numerical_columns         )], remainder='passthrough')


# In[ ]:


debug = False
print_column_values(features)


# In[ ]:


features_train_scale = ct.fit_transform(features_train)
features_test_scale  = ct.transform(features_test)
features_train_scale = pd.DataFrame(features_train_scale, columns = features_train.columns)
features_test_scale  = pd.DataFrame(features_test_scale, columns = features_test.columns)


# In[ ]:


debug = False
print_column_values(features_train_scale)


# In[ ]:


val_mse_list = []
val_mae_list = []
lvl_1_list   = [2**I for I in range(2,7)]
lvl_2_list   = [2**J for J in range(1,6)]
epochs_list  = [10*E for E in range(1,11)]
batch_list   = [2**J for J in range(1,5)]
print(lvl_1_list)
print(lvl_2_list)
print(epochs_list)
print(batch_list)
# MIN:  4 32 90       0.03884091228246689
lvl_1_list  = [32]
lvl_2_list  = [32]
epochs_list = [60]
batch_list  = [16]
for I in lvl_1_list:
    for J in lvl_2_list: 
        for E in epochs_list:
            for B in batch_list:
                print('I:J:E:B', I, J, E, B)
                unix1 = datetime.datetime.timestamp(datetime.datetime.now())*1000
                reset_seeds()
                model = Sequential()
                num_features = features_train_scale.shape[1]
                inputs   = InputLayer(input_shape=(num_features,))
                hidden1  = Dense(I, activation='relu')            
                dropout1 = Dropout(0.1)
                hidden2  = Dense(J, activation='relu')            
                dropout2 = Dropout(0.2)
                outputs  = Dense(1)
                model.add(inputs)
                model.add(hidden1)
                model.add(dropout1)
                model.add(hidden2)        
                model.add(dropout2)
                model.add(outputs)
                opt = Adam(learning_rate=0.01)
                model.compile(loss='mse', metrics=['mae'], optimizer=opt)
                stop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50)
                history = model.fit(features_train_scale, labels_train, epochs=E, batch_size=B, verbose=0, validation_split=0.2, callbacks=[stop])
                val_mse, val_mae = model.evaluate(features_test_scale, labels_test, verbose=0)
                unix2 = datetime.datetime.timestamp(datetime.datetime.now())*1000
                unixd = unix2 - unix1
                print(model.layers)
                print(model.summary())
                print(val_mse, val_mae)
                print('elapsed time:', unixd)
                val_mse_list.append([I, J, E, val_mse, unixd])
                val_mae_list.append([I, J, E, val_mae, unixd])


# In[ ]:


minI   = 0
minJ   = 0
minE   = 0
minMAE = 1
minT   = 0
for I,J,E,MAE,T in val_mae_list:
    # print(str(I), str(J), str(E), str(MSE), str(T))
    if MAE<minMAE:
        minI   = I
        minJ   = J
        minE   = E
        minMAE = MAE
        minT   = T
print('MIN:', str(minI), str(minJ), str(minE), str(minMAE), str(minT))


# In[ ]:


features_predict = pd.DataFrame(model.predict(features_test_scale))
features_predict.rename(columns={0:'COL'},inplace=True)
print(type(features_predict))
print(features_predict.iloc[:,0])
print(features_predict.columns)
features_predict_list = []
for index, row in features_predict.iterrows():
    features_predict_list.append(row['COL'])
print(features_predict_list[0:10])


# In[ ]:


labels_test = pd.DataFrame(labels_test)
labels_test.rename(columns={'Chance of Admit ':'COL'},inplace=True)
print(type(labels_test))
print(labels_test.iloc[:,0])
labels_test_list = []
for index, row in labels_test.iterrows():
    labels_test_list.append(row['COL'])
print(labels_test_list[0:10])


# In[ ]:


plt.plot(labels_test_list, features_predict_list, "o", alpha=0.5)
plt.show()


# In[ ]:


fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')
fig.tight_layout()
fig.savefig('plots.png')


# In[ ]:


predicted_values = model.predict(features_test_scale) 
print(r2_score(labels_test_list, features_predict_list)) 

