#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras


# In[21]:


dataset = pd.read_csv('breast_cancer.csv', header = None)


# In[22]:


dataset.head(5)


# In[23]:


data = dataset.iloc[:,:-1]
labels = dataset.iloc[:,-1]


# In[24]:


data


# In[25]:


labels


# In[26]:


data = data.replace('?', np.nan)


# In[27]:


data.iloc[235,:]


# In[28]:


data = data.fillna(0)


# In[29]:


data.iloc[235,:]


# In[30]:


from sklearn.preprocessing import normalize

data = normalize(data, axis = 0)


# In[31]:


data


# In[32]:


labels = np.array(labels)


# In[33]:


labels


# In[34]:


labels = np.where(labels ==2, 0, 1)
labels


# In[35]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train , y_test = train_test_split(data, labels, test_size = 0.20, random_state = 42)


# In[36]:


x_train.shape


# In[37]:


x_test.shape


# In[50]:


y_test


# In[40]:


from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping


# In[76]:


inputs = Input(shape =(9,))

for i in range(0,15):
    x = Dense(6)(inputs)
    x = Activation('relu')(x)
    x = Dense(4)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
outputs = Activation('sigmoid')(x)

model = Model(inputs, outputs)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =['accuracy'])


# In[77]:


model.summary()


# In[78]:


callback = EarlyStopping(monitor='val_loss', patience=50)
history = model.fit(x_train, y_train, epochs=300, batch_size=5, 
                    validation_data= (x_test, y_test), callbacks =callback, verbose =1)


# In[79]:


result = model.predict(x_test)


# In[80]:


result


# In[81]:


result = np.where(result>=0.5, 1, 0)
result


# In[64]:


result.shape


# In[82]:


result = result.reshape(result.shape[0])
result


# In[83]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print('accuracy = ',accuracy_score(y_test, result))


# In[84]:


pd.DataFrame(confusion_matrix(y_test, result))


# In[85]:


print(classification_report(y_test, result))


# In[86]:


print('biglearn.ir')


# In[ ]:




