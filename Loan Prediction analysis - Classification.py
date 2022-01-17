#!/usr/bin/env python
# coding: utf-8

# # Dataset Information
# 
#    Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers.
#    
#    This is a standard supervised classification task.A classification problem where we have to predict whether a loan would be approved or not. Below is the dataset attributes with description.
#    
# Variable | Description
# ----------|--------------
# Loan_ID | Unique Loan ID
# Gender | Male/ Female
# Married | Applicant married (Y/N)
# Dependents | Number of dependents
# Education | Applicant Education (Graduate/ Under Graduate)
# Self_Employed | Self employed (Y/N)
# ApplicantIncome | Applicant income
# CoapplicantIncome | Coapplicant income
# LoanAmount | Loan amount in thousands
# Loan_Amount_Term | Term of loan in months
# Credit_History | credit history meets guidelines
# Property_Area | Urban/ Semi Urban/ Rural
# Loan_Status | Loan approved (Y/N)

# ## Import modules

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# ## Loading the dataset

# In[3]:


df = pd.read_csv("Loan Prediction Dataset.csv")
df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# ## Preprocessing the dataset

# In[6]:


# find the null values
df.isnull().sum()


# In[7]:


# fill the missing values for numerical terms -  using mean
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())


# In[8]:


# fill the missing values for categorical terms - using mode(most frequently occurring value)
df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0]) # [0] bec we need only value 
df['Married'] = df["Married"].fillna(df['Married'].mode()[0])
df['Dependents'] = df["Dependents"].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df["Self_Employed"].fillna(df['Self_Employed'].mode()[0])


# In[9]:


df.isnull().sum()


# ## Exploratory Data Analysis

# In[10]:


# categorical attributes visualization
sns.countplot(df['Gender'])


# In[11]:


sns.countplot(df['Married'])


# In[12]:


sns.countplot(df['Dependents'])


# In[13]:


sns.countplot(df['Education'])


# In[14]:


sns.countplot(df['Self_Employed'])


# In[15]:


sns.countplot(df['Property_Area'])


# In[16]:


sns.countplot(df['Loan_Status']) # Approved and not approved


# In[ ]:





# In[17]:


# numerical attributes visualization
sns.distplot(df["ApplicantIncome"])


# In[18]:


sns.distplot(df["CoapplicantIncome"])


# In[19]:


sns.distplot(df["LoanAmount"])


# In[20]:


sns.distplot(df['Loan_Amount_Term'])


# In[21]:


sns.distplot(df['Credit_History'])


# In[ ]:





# ## Creation of new attributes

# In[22]:


# we can add Applicanticome + coapplicantincome to create total income of family because both belongs to same family
# total income
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.head()


# In[23]:


# As all the numerical attribute visualization is not a suitable distribution for training the model.
# Some of them are Left Skewed and some are right but none are normalized, So we have to use log transformation to normalize the data.
#Log transformation reduces or removes the Skewness of the original data
# we can also use min max normalization or standardization in order to train the model better.


# ## Log Transformation

# In[24]:


# apply log transformation to the attribute
df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome']+1)
sns.distplot(df["ApplicantIncomeLog"])


# In[25]:


df['CoapplicantIncomelog'] = np.log(df['CoapplicantIncome']+1)
sns.distplot(df["CoapplicantIncomelog"])
# Its not the best but its normalized to some extent


# In[26]:


df['LoanAmountLog'] = np.log(df['LoanAmount']+1)
sns.distplot(df["LoanAmountLog"])


# In[27]:


df['Loan_Amount_Term_Log'] = np.log(df['Loan_Amount_Term']+1)
sns.distplot(df["Loan_Amount_Term_Log"])
# its normalized to some extent the diffrence here in not in hundereds if you observe, it means it normalized data in some manner


# In[28]:


df['Total_Income_Log'] = np.log(df['Total_Income']+1)
sns.distplot(df["Total_Income_Log"])
# The new attribute which we created


# ## Coorelation Matrix

# In[29]:


# A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. 
corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot = True, cmap="BuPu")


# In[31]:


df.head()


# In[35]:


# drop unnecessary columns
cols = ['ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "Loan_Amount_Term", "Total_Income", 'Loan_ID', 'CoapplicantIncomelog']
df = df.drop(columns=cols, axis=1) # Axis = 0 will drop row wise and axis =1 drop column entirely
df.head()


# ## Label Encoding - To convert categorical attributes into numerical attributes to work 
# 

# In[36]:


# We used label encoder instead of One Hot encoder bec this has only 2 values.
from sklearn.preprocessing import LabelEncoder
cols = ['Gender',"Married","Education",'Self_Employed',"Property_Area","Loan_Status","Dependents"]
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])


# In[37]:


df.head()


# 
# 
# 
# 
# ## Train-Test Split

# In[38]:


# specify input and output attributes
X = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']


# In[222]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#random_state is basically used for reproducing your problem the same every time it is run. 


# ## Model Training

# In[223]:


# classify function
from sklearn.model_selection import cross_val_score
def classify(model, x, y): # Classify input and output attributes
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # Split the data
    model.fit(x_train, y_train) # After spliting we train the model
    print("Accuracy is", model.score(x_test, y_test)*100) # Print the accuracy # used 100 to display in percentage format
    # cross validation -use to define a data set to test the model in the training phase in order to limit problems like overfitting,underfitting and get an insight on how the model will generalize to an independent data set.
    # eg: cv-5, train-4, test-1
    score = cross_val_score(model, x, y, cv=5) # Cv = 5 is how many splits we want
    print("Cross validation is",np.mean(score)*100) # Score is 5


# In[224]:


# Going through all the basic Medels to get better accuracy 
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model, X, y)


# In[225]:


from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
model = RandomForestClassifier()
classify(model, X, y)


# In[226]:


model = ExtraTreesClassifier()
classify(model, X, y)


# In[227]:



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model, X, y)


# ## Hyperparameter tuning

# In[228]:


model =  RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=1)
classify(model, X, y)


# ## Confusion Matrix
# 
# A Confusion matrix will display a matrix with actaul labels and predicted lables and it will display the counts of how much it is correctly predicted, We can use this to reduce the errors also
# 
# First to use the confusion matrix we need to use the x_train , y_train(training data)and we will use the random forest model

# In[231]:


model = RandomForestClassifier()
model.fit(x_train, y_train)


# In[232]:


# Ploting confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = model.predict(x_test) # It will predict from the test data
cm = confusion_matrix(y_pred,y_test) # y_test is the actual value we have  from the data set and Y_pred is value we predicted from the model
cm


# In[233]:


sns.heatmap(cm, annot=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




