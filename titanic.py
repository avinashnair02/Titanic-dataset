#!/usr/bin/env python
# coding: utf-8

# TITANIC DATASET TRAINING........

# In[80]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[81]:


train=pd.read_csv('titanic_train.csv')


# In[82]:


train.head(100)


# In[83]:


train.isnull()


# In[84]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[85]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[7]:


sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[86]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[87]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=40)


# In[88]:


train['Age'].hist(bins=30,color='darkblue',alpha=0.3)


# In[89]:


sns.countplot(x='SibSp',data=train)


# In[90]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# DATA CLEANING....

# In[92]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[93]:


def impute_age(cols):
    Age = cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        
        if Pclass ==1:
            return 37
        elif Pclass ==2:
            return 29
        else:
            return 24
    else:
        return Age 
    


# In[94]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[95]:


sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis')


# In[96]:


train.drop('Cabin',axis=1,inplace=True)        


# In[97]:


train.head()


# In[98]:


sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis')


# In[99]:


train.dropna(inplace=True)


# CONVERTING CATERGORICAL FEATURES

# In[100]:


train.info()


# In[ ]:





# In[101]:


pd.get_dummies(train['Embarked'],drop_first=True).head()


# In[102]:


sex=pd.get_dummies(train['Sex'],drop_first=True)
embark=pd.get_dummies(train['Embarked'],drop_first=True)


# In[103]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[104]:


train.head()


# In[29]:


train=pd.concat([train,sex,embark],axis=1)


# In[105]:


train.head()


# BUILDING A LOGISTICS REGRESSION MODEL

# TRAIN TEST SPLIT

# In[33]:


train.drop('Survived',axis=1).head()


# In[106]:


train['Survived'].head()


# In[78]:


from sklearn.model_selection import train_test_split


# In[107]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# TRAINING AND PREDICTIONS

# In[108]:


from sklearn.linear_model import LogisticRegression


# In[109]:


logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)


# In[110]:


predictions=logmodel.predict(X_test)


# In[111]:


from sklearn.metrics import confusion_matrix


# In[112]:


accuracy =confusion_matrix(y_test,predictions)


# In[113]:


accuracy


# In[117]:


from sklearn.metrics import accuracy_score 


# In[118]:


accuracy=accuracy_score(y_test,predictions)


# In[119]:


accuracy


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




