#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[6]:


from sklearn.datasets import load_boston


# In[8]:


boston = load_boston()


# In[10]:


print(boston)


# In[11]:


# checking the key


boston.keys()


# In[14]:


# checking discription

print(boston.DESCR)


# In[15]:


# input feature

print(boston.data)


# In[16]:


# output feature

print(boston.target)


# In[17]:


print(boston.feature_names)


# ## Lets prepare the dataframe 

# In[18]:


dataset = pd.DataFrame(boston.data, columns=boston.feature_names)


# In[20]:


dataset.head()


# In[21]:


dataset['Price']=boston.target


# In[22]:


dataset.head()


# ## basic EDA

# In[23]:


dataset.info()


# In[24]:


dataset.describe()


# In[26]:


# check the moissing value

dataset.isnull().sum()


# #  in leanear regression  most important thing is the corelation of dependent and independent feature

# In[29]:


## EDA

dataset.corr()


# In[30]:


sns.pairplot(dataset)


# In[41]:


sns.set(rc={'figure.figsize':(8,8)})
sns.heatmap(dataset.corr(), annot=True)


# In[46]:


plt.scatter(dataset['CRIM'],dataset['Price'])
sns.set(rc={'figure.figsize':(8,6)})
plt.xlabel('Crim Rate')
plt.ylabel('Price')


# In[47]:


plt.scatter(dataset['RM'],dataset['Price'])


# In[48]:


sns.regplot(x='RM', y='Price',data = dataset)


# In[49]:


sns.regplot(x='LSTAT', y='Price',data = dataset)


# In[50]:


# outlier

sns.boxplot(dataset['Price'])


# In[52]:



# outlier
sns.boxplot(dataset['CRIM'])


# ## Independent and Dependent Feature
# 
#   - dependent feature is a series / single dimention array
#   - independent feature is a array / dataframe

# In[62]:



# assumed X is an indeppendent feature
X = dataset.iloc[:,:-1]  #getting all column name except "Price"

y = dataset.iloc[:,-1]   #getting only "Price" column


# In[60]:


X.head()


# In[61]:


y


# # data training and testing

# In[118]:


from sklearn.model_selection import train_test_split


# In[119]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=10)


# In[120]:


# independent feature

X_train


# In[121]:


X_test


# In[122]:


# dependent feature

y_train


# In[123]:


y_test


# In[124]:


X_train.shape


# In[125]:


y_train.shape


# In[126]:


X_test.shape


# In[127]:


y_test.shape


# In[ ]:





# # Standardization the dataset or Feature Scaling the dataset
# 
# - mean = 0
# - standerddeviation = 1

# In[128]:


from sklearn.preprocessing import StandardScaler


# In[129]:


scaler = StandardScaler()


# In[130]:


scaler


# In[131]:


# fit_transform = send the data and change the data

X_train = scaler.fit_transform(X_train)


# In[132]:


X_test = scaler.transform(X_test)   #to avoide the data leakage 


# # Model Training

# In[107]:


# multiple regressiom problem


# In[133]:


from sklearn.linear_model import LinearRegression


# In[134]:


regression = LinearRegression


# In[135]:


regression


# In[137]:


# fit = train the data / training the data 

regression.fit(X_train, y_train)


# In[117]:


y_train


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




