#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print('Data imported succesfully')
df.head(5)


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


plt.plot(df['Hours'], df['Scores'], linestyle='none', marker='o', color='r')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[6]:


X = df.iloc[:,:-1].values
X[0:10]


# In[7]:


y = df.iloc[:,1].values
y[0:10]


# In[8]:


sns.set_style('whitegrid') 
sns.lmplot(x ='Hours', y ='Scores', data = df)


# In[9]:


from sklearn.model_selection import train_test_split

#we split 80% of the data to the training set while 20% of the data to test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[10]:


print(X_train.shape) 
print(X_test.shape) 
  
# printing the shapes of the new y objects 
print(y_train.shape) 
print(y_test.shape)


# In[11]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)
lr.fit(X_train,y_train)
print('Training complete')


# In[12]:


print(lr.intercept_)

#For retrieving the slope:
print(lr.coef_)


# In[13]:


print(X_test)

#predicting the scores
y_pred = lr.predict(X_test)


# In[14]:


#comparing actual vs predicted

df_data = pd.DataFrame({'Actual': y_test,'Predicted':y_pred})
df_data


# In[15]:


df_data1 =df_data.head(5)
df_data1.plot(kind='bar',figsize=(15,9))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.show()


# In[16]:


plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# In[17]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[18]:


hours = 9.25
prediction = lr.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(prediction[0]))


# In[ ]:




