#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("kyphosis.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[7]:


sns.pairplot(df, hue = "Kyphosis", palette = "coolwarm")


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X = df.drop("Kyphosis", axis = 1)


# In[10]:


X


# In[11]:


y = df["Kyphosis"]


# In[12]:


y


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[21]:


from sklearn.tree import DecisionTreeClassifier


# In[22]:


dtree = DecisionTreeClassifier()


# In[23]:


dtree.fit(X_train, y_train)


# In[39]:


predictions = dtree.predict(X_test)


# In[40]:


predictions


# In[25]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[26]:


print(confusion_matrix(y_test, predictions))
print("\n")
print(classification_report(y_test, predictions))


# In[41]:


from sklearn.ensemble import RandomForestClassifier


# In[42]:


rfc = RandomForestClassifier(n_estimators=200)


# In[43]:


rfc.fit(X_train, y_train)


# In[44]:


rfc_pred = rfc.predict(X_test)


# In[45]:


rfc_pred


# In[46]:


print(confusion_matrix(y_test, rfc_pred))
print("\n")
print(classification_report(y_test, rfc_pred))


# In[49]:


df["Kyphosis"].value_counts()


# In[ ]:




