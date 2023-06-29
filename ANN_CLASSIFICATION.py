#!/usr/bin/env python
# coding: utf-8

# In[1]:


#we will try classification using artificial neural networks 


# In[2]:


import numpy as np 
import pandas as pd


# In[3]:


import tensorflow as tf
import keras 


# In[4]:


#loading the dataset 


# In[5]:


df = pd.read_csv("C:\\Users\\Sricharan Reddy\\Downloads\\Churn_Modelling.csv")


# In[6]:


df.head()


# In[7]:


#here exited column is the output column 


# In[8]:


df.info()


# In[9]:


df.drop(columns=['RowNumber','CustomerId','Surname'],inplace = True)


# In[10]:


df.head()


# In[11]:


#i have removed the rownumber,customerid,surname  columns because it is having no relation with the model by sense 


# In[12]:


#now encoding the data using dummy encoder 


# In[ ]:





# In[21]:


x = pd.get_dummies(df.drop(columns=['Exited']),drop_first=True)


# In[22]:


x


# In[23]:


#encoding is done 


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


y = df['Exited']


# In[28]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2,random_state=9)


# In[31]:


from sklearn.preprocessing import StandardScaler


# In[32]:


sc = StandardScaler()


# In[33]:


x_train = sc.fit_transform(x_train)


# In[34]:


x_test = sc.transform(x_test)


# In[ ]:





# In[29]:


#now lets build a deep learning model


# In[35]:


from keras.models import Sequential


# In[36]:


model = Sequential()


# In[37]:


from keras.layers import Dense


# In[54]:


model.add(Dense(input_dim=11,units=6,kernel_initializer='uniform',activation='relu'))


# In[55]:


model.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))


# In[56]:


model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))


# In[57]:


model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics = ['accuracy'])


# In[58]:


model.fit(x_train,y_train,epochs = 68,batch_size=32)


# In[59]:


from sklearn.metrics import accuracy_score


# In[60]:


y_pred_test = model.predict(x_test)


# In[61]:


y_pred_test = (y_pred_test  > 0.5)


# In[62]:


accuracy_score(y_test,y_pred_test)


# In[ ]:




