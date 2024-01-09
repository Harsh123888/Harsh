#!/usr/bin/env python
# coding: utf-8

# ### Importing required libraries

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### loading the data

# In[2]:


df=pd.read_csv(r"C:\Users\Harsh Gupta\Downloads\Admission_Prediction.csv")


# In[3]:


df


# In[4]:


df.head(10)


# ### Dropping irrelevant features

# In[5]:


df.drop('Serial No.',axis=1,inplace = True)


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


print("Number of Rows",df.shape[0])
print("Number of columns",df.shape[1])


# ### Checking the missing valuaes

# In[9]:


df.isnull().sum()


# ### Filling Missing Values

# In[10]:


df['GRE Score'].mean()


# In[11]:


df['TOEFL Score'].mean()


# In[12]:


df['University Rating'].mean()


# In[13]:


from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values=np.nan,strategy='mean')


# In[14]:


df[['GRE Score','TOEFL Score','University Rating']]=si.fit_transform(df[['GRE Score','TOEFL Score','University Rating']])


# In[15]:


df.head(20)


# In[16]:


df.isnull().sum()


# In[17]:


df.columns


# ### Checking the data type

# In[18]:


df.info()


# Obsv:
# 1. All the dataset is in numeric format
# 2.All the features have 500 observations as non null.so wew can perform outlier handling on this data easily
# 3.To confirm data can be cross check again for null,nan and outliers

# #### Checking no. of unique values for each columns

# In[19]:


df.nunique()


# #### Checking the statistics of data

# In[20]:


df.describe()


# In[21]:


#histogram


# In[22]:


import matplotlib.pyplot as plt
import pandas as pd
feature_names = ['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research','Chance of Admit']
df[feature_names].hist()
plt.tight_layout() 
plt.show()


# PAIRPLOT
# 
# A pairplot plot a pairwise relationships in a dataset. The pairplot function creates a grid of Axes such that each variable in data will by shared in the y-axis across a single row and in the x-axis across a single column.

# In[23]:


sns.pairplot(data=df)


# ### finding correlation`

# In[24]:


df.corr()


# HEATMAP
# 
# A heatmap is a graphical representation of data that uses a system of color-coding to represent different values. Heatmaps are used in various forms of analytics but are most commonly used to show user behaviour on specific webpages or webpage templates.

# In[25]:


plt.figure(figsize=(10,8))
sns.heatmap(data=df.corr(),annot=True,cmap='icefire')


# ### Box plot

# In[26]:


sns.boxplot(x="University Rating",y="GRE Score",data=df)


# ### Scatter plot

# In[27]:


feature_names = ['GRE Score', 'TOEFL Score', 'Chance of Admit']

# Create scatter plots for each feature against CGPA
for feature in feature_names:
    plt.figure(figsize=(4, 4))
    sns.scatterplot(x='CGPA', y=feature, data=df, color='m')
    plt.title(f'Scatter Plot: {feature} vs CGPA')
    plt.xlabel('CGPA')
    plt.ylabel(feature)
    plt.grid()
    plt.show()


# ### Linear Regression

# In[64]:


x = df.drop('Chance of Admit',axis=1)
y = df['Chance of Admit']


# In[29]:


from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import r2_score,accuracy_score,mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# In[30]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[31]:


from sklearn.linear_model import LinearRegression
linreg=LinearRegression()


# In[32]:


linreg.fit(X_train,y_train)


# In[33]:


ypred=linreg.predict(X_test)


# In[34]:


mae=mean_absolute_error(y_test,ypred)
mse=mean_squared_error(y_test,ypred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,ypred)


# In[35]:


print("mae:",mae)
print("mse:",mse)
print("rmse:",rmse)
print("r2:",r2)


# In[36]:


n=X_test.shape[0]
n


# In[37]:


num=(n-1)*(1-r2)


# In[38]:


p=X_test.shape[1]


# In[39]:


den=n-p-1


# In[40]:


re_adj=1-(num/den)


# In[41]:


re_adj


# In[42]:


df


# ### Polynomial Regression

# In[43]:


from sklearn.preprocessing import PolynomialFeatures


# In[44]:


pf = PolynomialFeatures(degree=2)
X_trainp=pf.fit_transform(X_train)
x_testp=pf.fit_transform(X_test)


# In[45]:


from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X_trainp,y_train)
y_pred=linreg.predict(x_testp)


# In[46]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)


# In[47]:


print("MAE : ",mae)
print("MSE : ",mse)
print("RMSE : ",rmse)
print("Accuracy of the model : ",r2)


# In[48]:


def c(GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research):
    feat=[[GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research]]
    y_pred = linreg.predict(x_testp)[0]
    print(y_pred)


# In[49]:


c(337,120,3,4,5,8,1)


# In[50]:


df


# In[51]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)


# In[52]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)


# In[53]:


from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))


# In[54]:


train = linreg.score(X_train, y_train)
test = linreg.score(X_test, y_test)

print(f"Training Results -: {train}")
print(f"Testing Results -: {test}")


# In[55]:


linreg.coef_


# In[56]:


from sklearn.linear_model import Ridge,Lasso
l2 = Ridge(alpha=1)
l2.fit(X_train,y_train)
y_pred = l2.predict(X_test)


# In[57]:


train = l2.score(X_train,y_train)
test = l2.score(X_test,y_test)

print(f"Training Results -: {train}")
print(f"Testing Results -: {test}")


# In[58]:


l2.coef_


# In[59]:


l1 = Lasso(alpha=0)
l1.fit(X_train,y_train)
y_pred = l1.predict(X_test)


# In[60]:


train = l1.score(X_train,y_train)
test = l1.score(X_test,y_test)

print(f"Training Results -: {train}")
print(f"Testing Results -: {test}")


# ### Hyperparameter Tuning

# In[61]:


l1=Lasso(alpha=0)
l1.fit(X_train,y_train)
y_pred = l1.predict(X_test)


# In[62]:


train = l1.score(X_train, y_train)
test = l1.score(X_test, y_test)

print(f"Training Results -: {train}")
print(f"Testing Results -: {test}")

