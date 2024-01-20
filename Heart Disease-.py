#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk


# ## Load the dataset

# In[3]:


df=pd.read_csv("Heart.csv")


# ## Analyze the dataset

# In[4]:


df.head()


# In[5]:


df.tail()


# ## Describe the dataset

# In[6]:


df.describe()


# ## Checking which columns contain null values

# In[7]:


df.isnull().sum()


# ## info()

# In[8]:


df.info()


# In[9]:


df.shape


# In[10]:


sns.countplot(data=df,x='Sex');
plt.title('Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')


# In[11]:


sns.displot(df['Age'])


# In[12]:


sns.displot(df['RestingBP'])


# In[13]:


sns.displot(df['Cholesterol'])


# ## Duplicate Values

# In[14]:


df_data=df.duplicated().any()


# In[15]:


df_data


# ### label

# In[16]:


df.HeartDisease.value_counts()


# # completing missing values & outlier

# In[17]:


from sklearn.impute import SimpleImputer
numImputer=SimpleImputer(missing_values=np.nan, strategy="most_frequent")
numImputer=numImputer.fit(df[["RestingBP","Cholesterol","FastingBS","MaxHR"]])
new_df=numImputer.transform(df[["RestingBP","Cholesterol","FastingBS","MaxHR"]])
new_df


# In[18]:


df[["RestingBP","Cholesterol","FastingBS","MaxHR"]]=new_df
df


# ### outlier

# In[19]:


plt.boxplot(x=[df["Cholesterol"],df["RestingBP"],df["MaxHR"],df["Oldpeak"]])
plt.xticks([1,2,3,4],["Cholesterol","RestingBP","MaxHR","Oldpeak"])
plt.show()


# In[20]:


plt.scatter(df["Cholesterol"],df["Age"])
plt.xlabel("Cholesterol")
plt.ylabel("Age")
plt.show()


# In[21]:


df = df.drop(df[(df['RestingBP'] == 0)].index)


# In[22]:


plt.boxplot(df["RestingBP"])
plt.show()


# In[23]:


df.loc[df['Cholesterol'] == 0,'Cholesterol'] = np.nan


# In[24]:


df.isnull().sum()


# In[25]:


df["Cholesterol"] = df["Cholesterol"].fillna(df["Cholesterol"].median())
df["Cholesterol"]


# # encoder

# #### Age

# In[26]:


df["Sex"].value_counts()


# In[27]:


df["Sex"]=df["Sex"].replace(["M","F"],[0,1])


# ####  ChestPainType

# In[28]:


df["ChestPainType"].value_counts()


# In[29]:


df["ChestPainType"]=df["ChestPainType"].replace(["ASY","NAP","ATA","TA"],[0,1,2,3])


# #### RestingECG

# In[30]:


df["RestingECG"].value_counts()


# In[31]:


df["RestingECG"]=df["RestingECG"].replace(["Normal","LVH","ST"],[0,1,2])


# #### ExerciseAngina

# In[18]:


df["ExerciseAngina"].value_counts()


# In[19]:


df["ExerciseAngina"]=df["ExerciseAngina"].replace(["N","Y"],[0,1])


# #### ST_Slope

# In[20]:


df["ST_Slope"].value_counts()


# In[21]:


df["ST_Slope"]=df["ST_Slope"].replace(["Flat","Up","Down"],[0,1,2])


# In[22]:


df


# ### corr

# In[31]:


cor=df.corr()
cor


# In[41]:


sns.pairplot(df)
plt.show()


# ## Standardization

# In[31]:


from sklearn import preprocessing
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]


# In[32]:


X


# In[33]:


Y


# # Split data

# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,Y, test_size=0.3 , random_state=0)
print("Train set",X_train.shape, y_train.shape)
print("Test set",X_test.shape, y_test.shape)


# In[35]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_s=sc.fit_transform(X_train)
X_test_s=sc.transform(X_test)


# In[36]:


X_train_s


# In[37]:


from sklearn.feature_selection import VarianceThreshold
variance_selector = VarianceThreshold(threshold=0)
X_train_fs = variance_selector.fit_transform(X_train_s)
X_test_fs = variance_selector.transform(X_test_s)
print(f"{X_train.shape[1]-X_train_fs.shape[1]} features have beenremoved, {X_train_fs.shape[1]} features remain")


# In[38]:


from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k='all')
X_train_fs1 = selector.fit_transform(X_train_s, y_train)
X_test_fs1 = selector.transform(X_test_s)
print(f"{X_train.shape[1]-X_train_fs1.shape[1]} features have beenremoved, {X_train_fs1.shape[1]} features remain")


# # KNN

# In[39]:


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=11, metric="manhattan")
classifier.fit(X_train_fs, y_train)


# In[386]:


y_pred=classifier.predict(X_test_fs)
print(y_pred)


# In[387]:


from sklearn import metrics
acc=metrics.accuracy_score(y_test, y_pred)
print("accuracy:%.2f\n\n"%(acc))
cm=metrics.confusion_matrix(y_test,y_pred)
print("confusion Matrix: ")
print(cm,"\n\n")
print("-------------------------------------------")
result=metrics.classification_report(y_test, y_pred)
print("Classification Report:\n")
print(result)


# ## other k

# In[362]:


for k in range(1, 12):
    classifier = KNeighborsClassifier(n_neighbors=k, metric="manhattan")
    classifier.fit(X_train_s, y_train)
    
    y_pred_train = classifier.predict(X_train_s) 
    accuracy_train = metrics.accuracy_score(y_train, y_pred_train)
    
    y_pred=classifier.predict(X_test_fs)
    accuracy_test = metrics.accuracy_score(y_test, y_pred)
    
    print(f"n_neighbors = {k},Train Accuracy = {accuracy_train:.2f}")
    print(f"n_neighbors = {k}, Test Accuracy = {accuracy_test:.2f}")
    print("")


# In[407]:


ax=sns.heatmap(cm, annot=True, fmt='d', cmap='flare')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()


# # Decision Tree

# In[40]:


from sklearn.tree import DecisionTreeClassifier
tree_classifier=DecisionTreeClassifier(criterion="gini", random_state=0)
tree_classifier.fit(X_train_fs, y_train)


# In[393]:


tree_y_pred=tree_classifier.predict(X_test_fs)
print(tree_y_pred)


# In[394]:


from sklearn import metrics
tree_acc=metrics.accuracy_score(y_test, tree_y_pred)
print("accuracy:%.2f\n\n"%(tree_acc))
tree_cm=metrics.confusion_matrix(y_test,tree_y_pred)
print("confusion Matrix: ")
print(tree_cm,"\n\n")
print("-------------------------------------------")
tree_result=metrics.classification_report(y_test, tree_y_pred)
print("Classification Report:\n")
print(tree_result)


# In[395]:


tree_ax=sns.heatmap(tree_cm, annot=True, fmt='d', cmap='flare')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')

plt.show()


# ## svm

# In[41]:


from sklearn import svm
svm_classifier=svm.SVC(kernel="rbf")
svm_classifier.fit(X_train_fs, y_train)


# In[46]:


svm_y_pred=svm_classifier.predict(X_test_fs)
print(svm_y_pred)


# In[47]:


from sklearn import metrics
svm_acc=metrics.accuracy_score(y_test, svm_y_pred)
print("accuracy:%.2f\n\n"%(svm_acc))
svm_cm=metrics.confusion_matrix(y_test,svm_y_pred)
print("confusion Matrix: ")
print(svm_cm,"\n\n")
print("-------------------------------------------")
svm_result=metrics.classification_report(y_test, svm_y_pred)
print("Classification Report:\n")
print(svm_result)


# In[48]:


svm_ax=sns.heatmap(svm_cm, annot=True, fmt='d', cmap='flare')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')

plt.show()

