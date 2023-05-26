#!/usr/bin/env python
# coding: utf-8

# # Diabetes Prediction

# ![download.jpg](attachment:download.jpg)

# # Content:
# Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
# 
# Pregnancies: Number of times pregnant,
# Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test,
# BloodPressure: Diastolic blood pressure (mm Hg),
# SkinThickness: Triceps skin fold thickness (mm),
# Insulin: 2-Hour serum insulin (mu U/ml),
# BMI: Body mass index (weight in kg/(height in m)^2),
# DiabetesPedigreeFunction: Diabetes pedigree function,
# Age: Age (years),
# Outcome: Class variable (0 or 1)

# # Import Libraries

# In[86]:


#Load libraries
#Data processing and Data Manipulation
import pandas as pd
#Calculation & array usage
import numpy as np
#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# # Load Data

# In[87]:


df = pd.read_csv("C:/Users/Talha Zafar/Downloads/diabetes.csv")
df


# In[88]:


df.head() #printed first five rows


# In[89]:


df.shape #checking number of rows and columns 


# # Data Preprocessing

# In[90]:


# checking for missing values
df.isnull().sum()


# In[91]:


#also used for checking missing values
df.isna().sum()


# In[92]:


df.info()


# In[93]:


df.describe() #summary of data


# In[94]:


df.corr()# to get correlation between features


# # Observation:
We observed that Glucose has maximum correlation with outcomes.
# #  Exploratory Data Analysis
# 

# In[95]:


df.hist(figsize=(20,20)) # ploting histogrom of each column in data frame


# # Ratio of patients suffering in diabetes

# In[96]:


df.Outcome.value_counts() 


# In[97]:


plt.figure(figsize=(5,4))
sns.countplot(x=df.Outcome)
plt.title("Patient Suffering in Diabetes")
plt.show()

We have two outcomes 0 and 1 , "0" means no patient is not diabetic and "1" means patient is diabetic.There are 500 of patients are not diabetic but 268 patients are diabetic.
# In[98]:


plt.pie(df.Outcome.value_counts(),labels = ["No","Yes"],autopct ="%.01f%%")
plt.legend(["No","Yes"])
plt.title('Patients Suffering in Diabetes')
plt.show()


# # Observation:
We analyze that approximately 35% of patients are diabetic.
# # Which Factors effecting diabetes?

# # Is age effect diabetes?

# In[99]:


plt.hist(df['Age'])


# In[100]:


df.Age.describe()


# In[101]:


#create age groups
bin_edges = [20, 40, 60, 80, 100]
df['Age_Group'] = pd.cut(df['Age'], bins=bin_edges)


# In[102]:


plt.figure(figsize=(7,7))
plt.pie(df['Age_Group'].value_counts(),labels = ["21 to 40","41 to 60","61 to 80","81 to 100"],autopct ="%.01f%%")
plt.title('Patient age groups')
plt.xlabel("Ages")
plt.legend(["21 to 40","41 to 60","61 to 80","81 to 100"])
plt.show()

There are approximately 75% of patients have age between 21 to 40 which is the highest ratio and the second highest ratio 22% are aged between 41 to 60.
# In[103]:


plt.figure(figsize=(7,6))
plt.xticks(rotation = 90)
sns.countplot(x=df['Age_Group'] , hue= df.Outcome)
plt.legend(['Non Diabetic','Diabetic'])
plt.show()


# # Observation:
We observed that age minorly affects diabetes. The patients aged between 40 to 60 are mostly diabetic. We know that 40 to 60 age is the start of old age and they are weak to weak with their age so that is why they are mostly diabetic.
# # Is glucose effect diabetes ?

# In[104]:


df.Glucose.describe()


# In[105]:


#create groups based on the waiting time column
bins = [0,25,50, 75 , 100 , 125 , 150 ,175 ,200]
df['Glucose_group'] = pd.cut(df['Glucose'], bins=bins)


# In[106]:


plt.figure(figsize=(8,8))
plt.pie(df['Glucose_group'].value_counts(),autopct ="%.01f%%")
plt.title('Glucose groups')
plt.legend(["0 to 25","26 to 50","51 to 75","76 to 100","101 to 125","126 to 150","151 to 175","176 to 200"])
plt.show()

There are approximately 34% of patients have glucose between 0 to 25.
# In[107]:


plt.figure(figsize=(8,6))
plt.xticks(rotation = 90)
sns.countplot(x= df['Glucose_group'], hue= df.Outcome)
plt.legend(['Non Diabetic','Diabetic'])

plt.show()


# # Observation:
Glucose is a factor for patients suffering from diabetes. The probability of diabetes increases when the patient's glucose level cross 110. From the given data set we also observed that the patients are mostly diabetic have a crossing glucose level of 110.
# # Is bloodpressure effect diabetes?

# In[108]:


df.BloodPressure.describe()


# In[109]:


#create age groups
bin_edge = [0,25,50, 75 , 100, 125]
df['BloodPressure_Group'] = pd.cut(df['BloodPressure'], bins=bin_edge)


# In[110]:


plt.figure(figsize=(8,8))
plt.pie(df['BloodPressure_Group'].value_counts(),autopct ="%.01f%%")
plt.title('Blood Pressure groups')
plt.legend(["0 to 25","26 to 50","51 to 75","76 to 100","101 to 125"])
plt.show()

There are mostly patients have blood pressure between 0 to 25.
# In[111]:


plt.figure(figsize=(8,6))
plt.xticks(rotation = 90)
sns.countplot(x= df['BloodPressure_Group'] , hue= df.Outcome)
plt.legend(['Non Diabetic','Diabetic'])
plt.show()


# # Observation:
Blood Pressure minorly affects diabetes while it is increasing the ratio of diabetic patients is increasing.
# # Is insulin effect diabetes?

# In[112]:


df.Insulin.describe()


# In[113]:


bine = [0,100 ,200 , 300 , 400 , 500 , 600 ,700 ,800 , 900]
df['Insulin_Group'] = pd.cut(df['Insulin'], bins=bine)


# In[114]:


plt.figure(figsize=(8,8))
plt.pie(df['Insulin_Group'].value_counts(),autopct ="%.01f%%")
plt.title('Insulin groups')
plt.legend(["0 to 100","101 to 200","201 to 300","301 to 400","401 to 500","501 to 600","601 to 700","701 to 800","801 to 900"])
plt.show()

There are mostly patients have glucose between 0 to 100 or 200.
# In[115]:


plt.figure(figsize=(16,4))
plt.xticks(rotation = 90)
sns.countplot(x= df['Insulin_Group'], hue= df.Outcome)
plt.legend(['Non Diabetic','Diabetic'])
plt.show()


# # Observation:
We observed from the above graph when the insulin amount increases the probability of diabetes increases.
# # Is pregnancies effect diabetes?

# In[116]:


plt.figure(figsize=(16,4))
plt.xticks(rotation = 90)
sns.countplot(x= df.Pregnancies , hue= df.Outcome)
plt.legend(['Non Diabetic','Diabetic'])
plt.show()


# # Observation:
We observed from the above graph when the number of pregnancies crosses 6 the probability of diabetes increases.
# # Comparison
# Blood Pressure with Age

# In[117]:


plt.scatter(df.Age,df.BloodPressure,color="red",alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("Comparison of Blood Pressure with Age")
plt.show()


# # Comparison
# Glucose with Age

# In[118]:


plt.scatter(df.Age,df.Glucose,alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.title('Comparison Glucose with Age')
plt.show()


# # Comparison
# Insulin with Glucose

# In[119]:


plt.scatter(df.Insulin,df.Glucose,color="purple",alpha=0.5)
plt.xlabel("Insulin")
plt.ylabel("Glucose")
plt.title("Comparison of Insulin with Glucose")
plt.show()


# # Trainning Data

# In[127]:


x=df[["Glucose"]]
y=df[["Outcome"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_train.shape
y_train.shape


# # Initlize the Classifier

# In[128]:


clf = MLPClassifier(hidden_layer_sizes=(51), solver="lbfgs", alpha=1e-5, activation="logistic")
clf.fit(x,y)
y_pred = clf.predict(x_train)


# # Check Accuracy as well as Confusion matrix

# In[129]:


print(accuracy_score(y_train,y_pred))
print(classification_report(y_train,y_pred))
print(confusion_matrix (y_train,y_pred))


# # Conclusion:
The accuracy of the model is 76.9%, which means that it correctly predicted the outcome of the classification task for about 77% of the data points in the training set. The classification report shows that the model has a higher precision and recall for the negative class (0) compared to the positive class (1), indicating that the model is better at predicting negative cases. The confusion matrix shows that out of 359 actual negative cases, the model correctly predicted 317 of them, while out of 178 actual positive cases, the model correctly predicted 96 of them. Overall, the model seems to perform reasonably well on the training set.
# In[ ]:


n = input("Enter glucose level:")
a = clf.predict(([[n]]))
if a==0:
    print("not diabetic")
else:
    print("diabetic")


# In[ ]:




