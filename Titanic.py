#!/usr/bin/env python
# coding: utf-8

# Titanic Problem - Using K-Neighbour Classification 
# 

# 1.Clearly,our aim is to predict whether the passenger survived or not . Binary Classification . 
# Hence, renameing the col= Survive as Label for better understanding. 
# 
# 2.Passenger Id is not required as a feature. 
# 
# 3.Need to include the number of family members, as it will affect who went on the lifeboat . 
#                 df['FamilyNos']=df[parch]+df[Sibch]
#                
# 4.Sex and age are TWO seperate entities , that shouldnt be combined
# eg: A male infant will have more chances than female adult .
# 
# 5.Embarked is not nesscary as their Destination was the same, all of them were on the boat . 
# 
# 6.Fare : Doubtful , I hope its per person . If its like family , it might depict something else. More fare , more rich,better chances of survival .
# 
# 7.Cabin : Not needed  .. Prolly later
# 

# In[1]:


#Adding required libraries
from sklearn import preprocessing,neighbors
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random


# In[2]:


df_train=pd.read_csv('C:\\Users\\dell\\Desktop\\Dataset\\Titanic\\train.csv')
df_test=pd.read_csv('C:\\Users\\dell\\Desktop\\Dataset\\Titanic\\test.csv')

df_train["Age"].fillna(-9999, inplace = True)
df_test["Age"].fillna(-9999, inplace = True)

df1=df_train.copy()
df2=df_test.copy()


print(len(df1))
print(len(df2))
print(df1.columns)
df1.head(500)


# By printining the table, its clear that few age rows are :NaN and few :Cabin's are Nan 
# Since, we are not incorporating Cabin, we only have to observe Age . 
# For this , Name column will be utilised . 
# 
# So, in case of AGE:NaN,wrt Name is there is : 
# Mr.-->(30,80)
# Mrs.-->(20,80)
# Miss.-->(0,19)
# Master.-->(0,29)
# 
# This way the data is not lost , and we can't afford to do so . 
# 
# def age_name()
# def clean_data()
# 
# Passenger_Id will not be dropped , until the training. 
# 
# Python Sorting Dataframe using a particular col val :
# 
# DataFrame.sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
# 
# Note: I'll put all NaN's at the end , and then slice it prolly and then work with it. 
# Nope,agian changes in data will be kastam . 
# 
# I want to use all 891 and 418 
# No dropping stuff.
# 

# In[3]:


#Data cleaning functions 

def mapp(name):
    if(name.find('Mr.')>0):
        nage=random.randrange(25,80)
    elif(name.find('Mrs.')>0):
        nage=random.randrange(20,80)   
    elif(name.find('Miss.')>0):
        nage=random.randrange(1,20)
    else:
        nage=random.randrange(1,25)
    return nage 


def age_name(df):
    A=df['Age']
    B=df['Name']
    df['Modified_Age']=[ ((x==-9999)*mapp(y))+ (x*(x!=-9999)) for (x,y) in zip(A,B) ]
    df.drop(['Age'],1,inplace=True)
    df.drop(['Name'],1,inplace=True)
    return df


## Superb , it works ! 
#Defining clean_data .

def clean_my_data(df):
    
    
    df['Family_nos']=df['SibSp']+df['Parch']
    df.drop(['SibSp'],1,inplace=True)
    df.drop(['Parch'],1,inplace=True)
 
    #Too many NAN for Cabin , better to remove it completely 
    df.drop(['Cabin'],1,inplace=True)
    df.drop([ 'Embarked'],1,inplace=True)
    df.drop(['Ticket'],1,inplace=True)
    df.head(5)
    
    #Replacing Male and Female , with tokens 
    L=dict({"male":0,"female":1})
    df['Sex']=[L[item] for item in df['Sex']]
    
    return df 


# In[4]:


#df1 Train data cleaning 

dff1=age_name(df1)
dfinal1=clean_my_data(dff1)
dfinal1.head()
print(len(dff1))
print(len(dfinal1))

#df2 Test data cleaning 

dff2=age_name(df2)
dfinal2=clean_my_data(dff2)
dfinal2.head()
print(len(dff2))
print(len(dfinal2))

print(dfinal1.columns)
print(dfinal2.columns)


# In[5]:


#Now cleaning part is done . 
#Training the damn thing .

W=dfinal1.drop('PassengerId',1)
W=W.drop('Survived',1)
#print(W.columns)

X=np.array(W)
y=np.array(dfinal1['Survived'])

Wq=dfinal2.copy()
Wq.dropna(inplace=True)

Plist=list(Wq['PassengerId'])

Wq=Wq.drop('PassengerId',1)
X_testi=np.array(Wq)


# In[6]:


clf1=neighbors.KNeighborsClassifier()

#Case 1 :

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.45)
clf1.fit(X_train,y_train)
acc=clf1.score(X_test,y_test)
print('Splitting the data set itself,in the ratio 0.45 :1')
print(acc)


# In[7]:


clf2=neighbors.KNeighborsClassifier()

#Case 2 :

clf2.fit(X,y)
z=clf2.predict(X_testi)
#print(z)
#print(len(z))


# In[8]:


#Now writing it dirctly to .csv file 
dictionary = dict({'PassengerId':Plist,'Survived':z})
#print(dictionary)
op=pd.DataFrame(dictionary)
op.head()


# In[9]:


#Now,writing onto the opfile.
fileop=open('titanic_op.txt','w')

T='PassengerId'
M='Survived'
fileop.write(str(T)+', '+str(M)+'\n')
for i in range(len(z)) :
    fileop.write(str(Plist[i])+','+str(y[i])+'\n')

fileop.close()
print('Fucking End')


# In[ ]:




