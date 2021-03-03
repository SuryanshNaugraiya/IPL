#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from six import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.naive_bayes import BernoulliNB , GaussianNB ,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LassoCV
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('IPL Matches 2008-2020.csv')
df = df[['team1' , 'team2' , 'venue' , 'toss_winner' ,'toss_decision' , 'winner' , 'neutral_venue']]


# In[3]:


df.head()


# In[4]:


df.team1.unique()


# In[5]:


df.venue.unique()


# In[6]:


# Keeping only consistent teams
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad' , 'Delhi Capitals']

consistent_venues = ['M Chinnaswamy Stadium',
       'Punjab Cricket Association Stadium, Mohali', 'Feroz Shah Kotla', 'Wankhede Stadium', 'Sawai Mansingh Stadium',
       'MA Chidambaram Stadium, Chepauk', 'Eden Gardens', 'Dr DY Patil Sports Academy', 'Brabourne Stadium',
       'Sardar Patel Stadium, Motera', 'Himachal Pradesh Cricket Association Stadium', 'Subrata Roy Sahara Stadium',
       'Rajiv Gandhi International Stadium, Uppal', 'Shaheed Veer Narayan Singh International Stadium',
       'JSCA International Stadium Complex', 'Barabati Stadium', 'Maharashtra Cricket Association Stadium',
       'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium', 'Punjab Cricket Association IS Bindra Stadium, Mohali',
       'M.Chinnaswamy Stadium', 'Holkar Cricket Stadium', 'Vidarbha Cricket Association Stadium, Jamtha', 'Nehru Stadium',
       'Saurashtra Cricket Association Stadium']


# In[7]:


df = df[(df['team1'].isin(consistent_teams)) & (df['team2'].isin(consistent_teams)) & (df['neutral_venue']==0) & (df['venue'].isin(consistent_venues))]


# In[8]:


df.drop('neutral_venue', inplace=True , axis=1)


# In[9]:


# delhi capitals and delhi daredevils are same team
df.team1.unique()


# In[10]:


df['team1'] = np.where(df.team1 == 'Delhi Daredevils' , 'Delhi Capitals' , df.team1)
df['team2'] = np.where(df.team2 == 'Delhi Daredevils' , 'Delhi Capitals' , df.team2)
df['toss_winner'] = np.where(df.toss_winner == 'Delhi Daredevils' , 'Delhi Capitals' , df.toss_winner)
df['winner'] = np.where(df.winner == 'Delhi Daredevils' , 'Delhi Capitals' , df.winner)


# In[11]:


# M Chinnaswamy Stadium and M.Chinnaswamy Stadium are same team
# Punjab Cricket Association IS Bindra Stadium, Mohali and Punjab Cricket Association Stadium, Mohali are same
df.venue.unique()


# In[12]:


df['venue'] = np.where(df.venue == 'M.Chinnaswamy Stadium' , 'M Chinnaswamy Stadium' , df.venue)
df['venue'] = np.where(df.venue == 'Punjab Cricket Association IS Bindra Stadium, Mohali' , 'Punjab Cricket Association Stadium, Mohali' , df.venue)


# In[13]:


df.venue.unique()


# In[14]:


df.head()


# In[15]:


def getNumber_team(x):
    if x=='Royal Challengers Bangalore':
        return 0
    elif x=='Kings XI Punjab':
        return 1
    elif x=='Delhi Capitals':
        return 2
    elif x=='Mumbai Indians':
        return 3     
    elif x=='Rajasthan Royals':
        return 4
    elif x=='Chennai Super Kings':
        return 5
    elif x=='Kolkata Knight Riders':
        return 6
    else:
        return 7


# In[16]:


df['team1'] = df['team1'].apply(getNumber_team)
df['team2'] = df['team2'].apply(getNumber_team)
df['toss_winner'] = df['toss_winner'].apply(getNumber_team)
df['winner'] = df['winner'].apply(getNumber_team)


# In[17]:


df.venue.unique()


# In[18]:


def getNumber_venue(x):
    if x=='M Chinnaswamy Stadium':
        return 1
    elif x=='Punjab Cricket Association Stadium, Mohali':
        return 2
    elif x=='Feroz Shah Kotla':
        return 3
    elif x=='Wankhede Stadium':
        return 4
    elif x=='Sawai Mansingh Stadium':
        return 5
    elif x=='MA Chidambaram Stadium, Chepauk':
        return 6
    elif x=='Eden Gardens':
        return 7
    elif x=='Dr DY Patil Sports Academy':
        return 8
    elif x=='Brabourne Stadium':
        return 9
    elif x=='Sardar Patel Stadium, Motera':
        return 10 
    elif x=='Himachal Pradesh Cricket Association Stadium':
        return 11
    elif x=='Subrata Roy Sahara Stadium':
        return 12
    elif x=='Rajiv Gandhi International Stadium, Uppal':
        return 13
    elif x=='Shaheed Veer Narayan Singh International Stadium':
        return 14
    elif x=='JSCA International Stadium Complex':
        return 15
    elif x=='Maharashtra Cricket Association Stadium':
        return 16
    elif x=='Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium':
        return 17
    elif x=='Barabati Stadium':
        return 18
    else:
        return 19


# In[19]:


df['venue'] = df['venue'].apply(getNumber_venue)


# In[20]:


def getNumber_tossDecision(x):
    if x=='field':
        return 0
    else:
        return 1


# In[21]:


df['toss_decision'] = df['toss_decision'].apply(getNumber_tossDecision)


# In[22]:


df.dtypes


# In[23]:


df


# In[24]:


df.corr()


# In[25]:


import seaborn as sns
sns.heatmap(df.corr())


# In[26]:


X = df.drop(labels='winner', axis=1)
y = df['winner']

X = np.array(X)
y = np.array(y)

for i in range(len(y)):
    if y[i]==X[i][0]:
        y[i]=0
    else:
        y[i]=1
        
# imbalanced dataset so it is going to 
print(np.unique(y, return_counts=True))
# In[27]:


zeros = 0
for i in range(len(X)):
    if y[i] == X[i][0]:
        if zeros <= 250:
            y[i] = 0
            zeros = zeros + 1
        else:
            y[i] = 1
            t = X[i][0]
            X[i][0] = X[i][1] 
            X[i][1] = t
    else:
        y[i] = 1
        

for i in range(len(X)):
    if X[i][3]==X[i][0]:
        X[i][3]=0
    else:
        X[i][3]=1


# In[28]:


X = np.array(X , dtype='int32')
y = np.array(y , dtype='int32')

y = y.ravel()

print(np.unique(y, return_counts=True))
# now balanced dataset


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2 , random_state=0)


# In[30]:


alg1 = LogisticRegression(solver='liblinear')

start = time.time()
alg1.fit(X_train , y_train)
end = time.time()
total_time1 = end - start

y_pred1 = alg1.predict(X_test)

print('accuracy : ', alg1.score(X_test , y_test))
print('time : ' , total_time1)
print(classification_report(y_test , y_pred1))
print(confusion_matrix(y_test , y_pred1))


# In[31]:


alg2 = RandomForestClassifier(n_estimators=60)

start = time.time()
alg2.fit(X_train , y_train)
end = time.time()
total_time2 = end - start

y_pred2 = alg2.predict(X_test)

print('accuracy : ', alg2.score(X_test , y_test))
print('time : ' , total_time2)
print(classification_report(y_test , y_pred2))
print(confusion_matrix(y_test , y_pred2))


# In[32]:


alg3 = DecisionTreeClassifier(max_depth=1 , criterion='gini')

start = time.time()
alg3.fit(X_train , y_train)
end = time.time()
total_time3 = end - start

y_pred3 = alg3.predict(X_test)

print('accuracy : ', alg3.score(X_test , y_test))
print('time : ' , total_time3)
print(classification_report(y_test , y_pred3))
print(confusion_matrix(y_test , y_pred3))


# In[33]:


# Printing tree alongwith class names
dot_data = StringIO()
export_graphviz(alg3, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names = ['team1', 'team2', 'venue','toss_winner', 'toss_decision'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())


# In[34]:


dot_data = export_graphviz(alg3, out_file=None,
                          feature_names=['team1', 'team2', 'venue', 'toss_winner', 'toss_decision'])
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("ipl_winner_decision_tree.pdf")


# In[35]:


alg4 = BernoulliNB()

start = time.time()
alg4.fit(X_train,y_train)
end = time.time()
total_time4 = end - start

y_pred4 = alg4.predict(X_test)

print('accuracy : ', alg4.score(X_test , y_test))
print('time : ' , total_time4)
print(classification_report(y_test , y_pred4))
print(confusion_matrix(y_test , y_pred4))


# In[36]:


alg5 = GaussianNB()

start = time.time()
alg5.fit(X_train,y_train)
end = time.time()
total_time5 = end - start

y_pred5 = alg5.predict(X_test)

print('accuracy : ', alg5.score(X_test , y_test))
print('time : ' , total_time5)
print(classification_report(y_test , y_pred5))
print(confusion_matrix(y_test , y_pred5))


# In[37]:


alg6 = MultinomialNB()

start = time.time()
alg6.fit(X_train,y_train)
end = time.time()
total_time6 = end - start

y_pred6 = alg6.predict(X_test)

print('accuracy : ', alg6.score(X_test , y_test))
print('time : ' , total_time6)
print(classification_report(y_test , y_pred6))
print(confusion_matrix(y_test , y_pred6))


# In[38]:


x_axis = []
y_axis = []
for k in range(1, 26, 2):
    clf = KNeighborsClassifier(n_neighbors = k)
    score = cross_val_score(clf, X_train, y_train, cv = KFold(n_splits=5, shuffle=True, random_state=0))
    x_axis.append(k)
    y_axis.append(score.mean())


# In[39]:


import matplotlib.pyplot as plt
plt.plot(x_axis, y_axis)
plt.xlabel("k")
plt.ylabel("cross_val_score")
plt.title("variation of score on different values of k")
plt.show()


# In[40]:


alg7 = KNeighborsClassifier(n_neighbors=19, weights='distance', algorithm='auto', p=2, metric='minkowski')

start = time.time()
alg7.fit(X_train, y_train)
end = time.time()
total_time7 = end - start

y_pred7 = alg7.predict(X_test)

print('accuracy : ', alg7.score(X_test , y_test))
print('time : ' , total_time7)
print(classification_report(y_test , y_pred7))
print(confusion_matrix(y_test , y_pred7))


# In[41]:


clf = SVC(kernel='rbf')
grid = {'C': [1e2,1e3, 5e3, 1e4, 5e4, 1e5],
       'gamma': [1e-3, 5e-4, 1e-4, 5e-3]}

alg8 = GridSearchCV(clf, grid)

start = time.time()
alg8.fit(X_train, y_train)
end = time.time()
total_time8 = end - start

y_pred8 = alg8.predict(X_test)

print(alg8.best_estimator_)

print('accuracy : ', alg8.score(X_test , y_test))
print('time : ' , total_time8)
print(classification_report(y_test , y_pred8))
print(confusion_matrix(y_test , y_pred8))


# In[42]:


alg9 = LinearSVC(multi_class='crammer_singer')

start = time.time()
alg9.fit(X_train, y_train)
end = time.time()
total_time9 = end - start

y_pred9 = alg9.predict(X_test)

print('accuracy : ', alg9.score(X_test , y_test))
print('time : ' , total_time9)
print(classification_report(y_test , y_pred9))
print(confusion_matrix(y_test , y_pred9))


# In[43]:


ridge = RidgeClassifier()

parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
alg10=GridSearchCV(ridge,parameters)

start = time.time()
alg10.fit(X_train, y_train)
end = time.time()
total_time10 = end - start

y_pred10 = alg10.predict(X_test)

print('accuracy : ', alg10.score(X_test , y_test))
print('time : ' , total_time10)
print(classification_report(y_test , y_pred10))
print(confusion_matrix(y_test , y_pred10))


# In[44]:


test = np.array([2, 4, 1, 1, 1]).reshape(1,-1)

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


# In[45]:


test = np.array([4, 2, 1, 0, 1]).reshape(1,-1)

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


# In[46]:


df_model=pd.DataFrame({
'Model_Applied':['Logistic_Regression', 'Random_Forest', 'Decision_tree', 'BernoulliNB', 'GausianNB', 'MultinomialNB', 'KNN', 'SVC', 'Linear_SVC', 'Ridge_Classifier'],
'Accuracy':[alg1.score(X_test,y_test), alg2.score(X_test,y_test), alg3.score(X_test,y_test), alg4.score(X_test,y_test),
            alg5.score(X_test,y_test), alg6.score(X_test,y_test), alg7.score(X_test,y_test), alg8.score(X_test,y_test),
            alg9.score(X_test,y_test), alg10.score(X_test,y_test)],
'Training_Time':[total_time1, total_time2, total_time3, total_time4, total_time5, total_time6, total_time7, total_time8, 
                 total_time9, total_time10]})


# In[47]:


df_model


# In[48]:


df_model.plot(kind='bar',x='Model_Applied', ylim=[0,1] , y='Accuracy', figsize=(10,10) , ylabel='Accuracy', title='Accurcy comparison of different Models')


# In[49]:


df_model.plot(kind='bar',x='Model_Applied', ylim=[0,0.14] , y='Training_Time', figsize=(10,10), ylabel='Training Time', title='Training time comparison of different Models')


# In[50]:


import pickle as pkl

with open('winner.pkl', 'wb') as f:
    pkl.dump(alg3, f)

with open('winner.pkl', 'rb') as f:
    model = pkl.load(f)

model.predict(test)

