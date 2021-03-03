#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import pydotplus
from six import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import GammaRegressor

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('ipl.csv')


# In[3]:


df.drop(columns=['mid','striker','non-striker','batsman','bowler'], inplace=True)
df.head()


# In[4]:


df.bat_team.unique()


# In[5]:


# Keeping only consistent teams
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']


# In[6]:


df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]


# In[7]:


df['bat_team'] = np.where(df.bat_team == 'Delhi Daredevils' , 'Delhi Capitals' , df.bat_team)
df['bowl_team'] = np.where(df.bowl_team == 'Delhi Daredevils' , 'Delhi Capitals' , df.bowl_team)


# In[8]:


df['venue'] = np.where(df.venue == '''St George's Park''' , 'St George Park' , df.venue)


# In[9]:


df.venue.unique()


# In[10]:


consistent_venues = ['M Chinnaswamy Stadium',
       'Punjab Cricket Association Stadium, Mohali', 'Feroz Shah Kotla', 'Wankhede Stadium', 'Sawai Mansingh Stadium',
       'MA Chidambaram Stadium, Chepauk', 'Eden Gardens', 'Dr DY Patil Sports Academy', 'Brabourne Stadium',
       'Sardar Patel Stadium, Motera', 'Himachal Pradesh Cricket Association Stadium', 'Subrata Roy Sahara Stadium',
       'Rajiv Gandhi International Stadium, Uppal', 'Shaheed Veer Narayan Singh International Stadium',
       'JSCA International Stadium Complex', 'Barabati Stadium', 'Maharashtra Cricket Association Stadium',
       'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium', 'Punjab Cricket Association IS Bindra Stadium, Mohali',
       'Holkar Cricket Stadium', 'Vidarbha Cricket Association Stadium, Jamtha', 'Nehru Stadium',
       'Saurashtra Cricket Association Stadium']


# In[11]:


df = df[(df['venue'].isin(consistent_venues))]


# In[12]:


df['venue'] = np.where(df.venue == 'Punjab Cricket Association IS Bindra Stadium, Mohali' , 'Punjab Cricket Association Stadium, Mohali' , df.venue)


# In[13]:


df.venue.unique()


# In[14]:


df = df[df.overs >= 5.0]


# In[15]:


df.head(10)


# In[16]:


df.dtypes


# In[17]:


# Converting the column 'date' from string into datetime object
from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))


# In[18]:


df.dtypes


# In[19]:


# --- Data Preprocessing ---
# Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=df, columns=['venue' , 'bat_team', 'bowl_team'])


# In[20]:


encoded_df.columns


# In[21]:


# rearranging order of encoded_df
encoded_df = encoded_df[['date', 
         
       'bat_team_Chennai Super Kings', 'bat_team_Delhi Capitals','bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
       'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
       'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
         
       'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Capitals',
       'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
       'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
       'bowl_team_Royal Challengers Bangalore',
       'bowl_team_Sunrisers Hyderabad',
         
       'venue_Barabati Stadium', 'venue_Brabourne Stadium',
       'venue_Dr DY Patil Sports Academy',
       'venue_Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
       'venue_Eden Gardens', 'venue_Feroz Shah Kotla',
       'venue_Himachal Pradesh Cricket Association Stadium',
       'venue_Holkar Cricket Stadium',
       'venue_JSCA International Stadium Complex',
       'venue_M Chinnaswamy Stadium', 'venue_MA Chidambaram Stadium, Chepauk',
       'venue_Maharashtra Cricket Association Stadium',
       'venue_Punjab Cricket Association Stadium, Mohali',
       'venue_Rajiv Gandhi International Stadium, Uppal',
       'venue_Sardar Patel Stadium, Motera', 'venue_Sawai Mansingh Stadium',
       'venue_Shaheed Veer Narayan Singh International Stadium',
       'venue_Subrata Roy Sahara Stadium', 'venue_Wankhede Stadium',
         
       'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'total']]


# In[22]:


df.corr()


# In[23]:


import seaborn as sns
sns.heatmap(df.corr())


# In[24]:


# Splitting the data into train and test set
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]


# In[25]:


y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values


# In[26]:


# Removing the 'date' column
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)


# In[27]:


alg1 = LinearRegression()

start = time.time()
alg1.fit(X_train , y_train)
end = time.time()
total_time1 = end - start

y_pred1 = alg1.predict(X_test)

print('accuracy : ', alg1.score(X_test , y_test))
print('time : ' , total_time1)


# In[28]:


alg2 = RandomForestRegressor(n_estimators=60)

start = time.time()
alg2.fit(X_train , y_train)
end = time.time()
total_time2 = end - start

y_pred2 = alg2.predict(X_test)

print('accuracy : ', alg2.score(X_test , y_test))
print('time : ' , total_time2)


# In[29]:


alg3 = DecisionTreeRegressor(max_depth=6)
start = time.time()
alg3.fit(X_train , y_train)
end = time.time()
total_time3 = end - start

y_pred3 = alg3.predict(X_test)

print('accuracy : ', alg3.score(X_test , y_test))
print('time : ' , total_time3)


# In[30]:


# Printing tree alongwith class names
dot_data = StringIO()
export_graphviz(alg3, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())


# In[31]:


dot_data = export_graphviz(alg3, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("ipl_score_prediction_tree.pdf")


# In[32]:


alg4 = TheilSenRegressor()

start = time.time()
alg4.fit(X_train , y_train)
end = time.time()
total_time4 = end - start

y_pred4 = alg4.predict(X_test)

print('accuracy : ', alg4.score(X_test , y_test))
print('time : ' , total_time4)


# In[33]:


alg5 = HuberRegressor()

start = time.time()
alg5.fit(X_train , y_train)
end = time.time()
total_time5 = end - start

y_pred5 = alg5.predict(X_test)

print('accuracy : ', alg5.score(X_test , y_test))
print('time : ' , total_time5)


# In[34]:


alg6 = GammaRegressor()

start = time.time()
alg6.fit(X_train , y_train)
end = time.time()
total_time6 = end - start

y_pred6 = alg6.predict(X_test)

print('accuracy : ', alg6.score(X_test , y_test))
print('time : ' , total_time6)


# In[35]:


x_axis = []
y_axis = []

for k in range(1, 30, 2):
    clf = KNeighborsRegressor(n_neighbors = k)
    score = cross_val_score(clf, X_train, y_train, cv = KFold(n_splits=5))
    x_axis.append(k)
    y_axis.append(score.mean())

plt.plot(x_axis, y_axis)
plt.xlabel("k")
plt.ylabel("cross_val_score")
plt.title("variation of score on different values of k")
plt.show()


# In[36]:


alg7 = KNeighborsRegressor(n_neighbors=25, weights='distance', algorithm='auto', p=2, metric='minkowski')

start = time.time()
alg7.fit(X_train, y_train)
end = time.time()
total_time7 = end - start

y_pred7 = alg7.predict(X_test)

print('accuracy : ', alg7.score(X_test , y_test))
print('time : ' , total_time7)


# In[37]:


alg8 = SVR()

start = time.time()
alg8.fit(X_train, y_train)
end = time.time()
total_time8 = end - start

y_pred8 = alg8.predict(X_test)

print('accuracy : ', alg8.score(X_test , y_test))
print('time : ' , total_time8)


# In[38]:


alg9 = LinearSVR()

start = time.time()
alg9.fit(X_train, y_train)
end = time.time()
total_time9 = end - start

y_pred9 = alg9.predict(X_test)

print('accuracy : ', alg9.score(X_test , y_test))
print('time : ' , total_time9)


# In[39]:


ridge = Ridge()

parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
alg10=GridSearchCV(ridge,parameters)

start = time.time()
alg10.fit(X_train, y_train)
end = time.time()
total_time10 = end - start

y_pred10 = alg10.predict(X_test)

print('accuracy : ', alg10.score(X_test , y_test))
print('time : ' , total_time10)


# In[40]:


test = np.array([1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,60,3,7,40,2]).reshape(1,-1)

print('alg1 : ' , alg1.predict(test))
print('alg2 : ' , alg2.predict(test))
print('alg3 : ' , alg3.predict(test))
print('alg4 : ' , alg4.predict(test))
print('alg5 : ' , alg5.predict(test))
print('alg6 : ' , alg6.predict(test))
print('alg7 : ' , alg7.predict(test))
print('alg8 : ' , alg8.predict(test))
print('alg9 : ' , alg9.predict(test))
print('alg10 :' , alg10.predict(test))


# In[41]:


df_model=pd.DataFrame({
'Model_Applied':['Linear_Regression', 'Random_Forest', 'Decision_tree', 'TheilSen_Regression', 'Huber_Regression', 
                 'Gamma_regreesor', 'KNN', 'SVR', 'Linear_SVR', 'Ridge_Regreesion'],
'Accuracy':[alg1.score(X_test,y_test), alg2.score(X_test,y_test), alg3.score(X_test,y_test), alg4.score(X_test,y_test),
            alg5.score(X_test,y_test), alg6.score(X_test,y_test), alg7.score(X_test,y_test), alg8.score(X_test,y_test),
            alg9.score(X_test,y_test), alg10.score(X_test,y_test)],
'Training_Time':[total_time1, total_time2, total_time3, total_time4, total_time5, total_time6, total_time7, total_time8, 
                 total_time9, total_time10]})


# In[42]:


df_model


# In[43]:


df_model.plot(kind='bar',x='Model_Applied', ylim=[0,1] , y='Accuracy', figsize=(10,10) , ylabel='Accuracy', title='Accurcy comparison of different Models')


# In[44]:


df_model.plot(kind='bar',x='Model_Applied', ylim=[0,12] , y='Training_Time', figsize=(10,10), ylabel='Training Time', title='Training time comparison of different Models')


# In[45]:


import pickle as pkl

with open('score.pkl', 'wb') as f:
    pkl.dump(alg3, f)

with open('score.pkl', 'rb') as f:
    model = pkl.load(f)

model.predict(test)

