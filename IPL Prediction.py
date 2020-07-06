#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing essential libraries
import pandas as pd
import pickle


# In[2]:


df = pd.read_csv('C:/Users/SAM/OneDrive/Desktop/Project/Kaggle Proj/IPL/IPL-First-Innings-Score-Prediction-Deployment-master/ipl.csv')


# In[3]:


df.head()


# In[4]:


# --- Data Cleaning ---
# Removing unwanted columns
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
df.drop(labels=columns_to_remove, axis=1, inplace=True)


# In[5]:


df.head()


# In[6]:


df['bat_team'].unique()


# In[7]:


# Keeping only consistent teams
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']


# In[8]:


df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]


# In[9]:



df.head()


# In[10]:


# Removing the first 5 overs data in every match
df = df[df['overs']>=5.0]


# In[11]:


df.head()


# In[12]:


print(df['bat_team'].unique())
print(df['bowl_team'].unique())


# In[13]:


# Converting the column 'date' from string into datetime object
from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))


# In[14]:


# --- Data Preprocessing ---
# Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])


# In[15]:


encoded_df.head()


# In[16]:


encoded_df.columns


# In[17]:


# Rearranging the columns
encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]


# In[18]:


# Splitting the data into train and test set
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]


# In[19]:


y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values


# In[20]:


# Removing the 'date' column
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)


# In[21]:


# --- Model Building ---
# Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[22]:


# Creating a pickle file for the classifier
filename = 'first-innings-score-lr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))


# In[23]:


## Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[24]:


ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)


# In[25]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[26]:


prediction=ridge_regressor.predict(X_test)


# In[27]:


import seaborn as sns
sns.distplot(y_test-prediction)


# In[28]:


from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[29]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[30]:


lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X_train,y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[31]:


predictionl=lasso_regressor.predict(X_test)


# In[33]:


import seaborn as sns
sns.distplot(y_test-predictionl)


# In[32]:


from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, predictionl))
print('MSE:', metrics.mean_squared_error(y_test, predictionl))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictionl)))


# In[ ]:




