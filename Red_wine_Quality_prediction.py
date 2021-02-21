#!/usr/bin/env python
# coding: utf-8

# ### Importing the necessary libraries

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot  as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading and Exploring the data

# In[2]:


df=pd.read_csv("D:\Misc\Projects_ML\winequality-red.csv")


# In[3]:


df['quality'].value_counts()


# In[4]:


df.shape


# In[5]:


df.describe()


# ###### Checking if the there are any null values present

# In[6]:


df.isnull().sum()


# ### Data Visualization

# In[7]:


sns.pairplot(df)
plt.show()
plt.figure(figsize=(30,20))


# In[8]:


corr = df.corr()
plt.figure(figsize=(30,20))
sns.heatmap(corr, annot=True, cmap='Blues')
b, t = plt.ylim()
plt.ylim(b+0.5, t-0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# ###### From correlation plot,alcohol has 0.48 correlation and sulphates has 0.25 correlation with Quality

# ###### Therefore,relationship for Alcohol and Quality with a scatter plot

# In[9]:


plt.scatter(df['alcohol'],df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')


# ###### Similarly,Relationship for Sulphates and Quality with a scatter plot

# In[10]:


plt.scatter(df['sulphates'],df['quality'])
plt.xlabel('sulphates')
plt.ylabel('quality')


# In[11]:


df['quality'].value_counts()


# ###### Representation of different qualities  in pie chart

# In[12]:


fig = plt.figure(figsize = (10,20))

plt.pie(df.quality.value_counts(), 
        autopct='%1.1f%%',
        labels=df.quality.unique());


# In[13]:


df['sulphates'].nunique()


# In[14]:


plt.hist(df.quality,bins=6,histtype='bar')

plt.title(' Quality Distribution')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()


# ###### The amount of alcohol is almost directly proportional to the quality

# In[15]:


sns.barplot(df.quality, df.alcohol, data=df )
plt.title(" Quality vs Alcohol")
plt.show()


# ###### Sulphates level is directly proportional to quality,From the bar graph,it is interpretted that as the level of sulphates goes up,the quality of the wine goes up

# In[16]:


sns.barplot(x = 'quality', y = 'sulphates', data = df)


# ###### The quality of the wine is split to be good and bad,quality is bad (0) if the value is 3,4,5,quality is good (1) if the value is 6,7,8.

# In[17]:


df['quality']=df['quality'].apply(lambda x:1 if x>5 else 0)


# In[18]:


fig = plt.figure(figsize = (10,6))

sns.countplot(x = 'quality', data = df)


# ### Preprocessing of the data

# In[19]:


X = df.iloc[:,:-1]        
Y = df.iloc[:,-1]          


# In[20]:


X.head()


# In[21]:


Y.head()


# ### Splitting the data set into train and test

# In[22]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# ###### Scaling

# In[23]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ### Building the Model

# In[24]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# ###### In machine learning, classification is general technique used for predicting the label for a given example of input data,Logistic Regression is most generally used for classication.However,for complex models,good accuracy is not always guaranteed.Therefore , decision tree classifier has been taken into account.
# 

# In deicision Tree classifier,intially it starts with a node with particular attribute from the data.Based on the outcome
# of test,there are branches.Finally, the  terminal nodes predict the class or outcome distribution.

# In[25]:


model_1=DecisionTreeClassifier(max_depth = 10)
model_1.fit(X_train, y_train)
pred_train_1=model_1.predict(X_train)


# In[26]:


from sklearn.metrics import accuracy_score
accuracy_score(y_train,pred_train_1)


# In[27]:


pred_test_1=model_1.predict(X_test)


# In[28]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred_test_1)


# ###### While decision tree classifier has moderate performance.With random forest tree classifier, the performance can be further improved.Hence, RandomForestClassifier has been further taken into account.

# This is mainly due to fact that ,multiple decision trees are randomly sampled(n_estimators) and based on the different decision
# tree classifiers, the final outcome is considered.

# In[29]:


model_2=RandomForestClassifier(n_estimators=19)
model_2.fit(X_train, y_train)
pred_train_2=model_2.predict(X_train)


# In[30]:


pred_test_2=model_2.predict(X_test)


# In[31]:


from sklearn.metrics import accuracy_score
accuracy_score(y_train,pred_train_2)


# In[32]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred_test_2)


# In[33]:


from sklearn.metrics import accuracy_score,classification_report
print("Classification report for Decision Tree Classifier is\n",classification_report(y_test,pred_test_1))


# In[34]:


from sklearn.metrics import accuracy_score,classification_report
print("Classification report for RandomForestClassifier is\n",classification_report(y_test,pred_test_2))


# ###### Conclusion:   From the results,it can be seen that RandomForestClassifier has better perfomance.Hence,Random forest classifier is considered further for this problem.
#     

# ###### Referneces:
# 1.https://medium.com/swlh/decision-tree-classification-de64fc4d5aac
