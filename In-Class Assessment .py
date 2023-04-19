#!/usr/bin/env python
# coding: utf-8

# # Multiple-Choice Questions:
# 
# 1.What is Linear Regression? 
# 
# b. A supervised learning algorithm used for regression problems.
# 
# 2.Which of the following is NOT an assumption of Linear Regression? 
# 
# c. Multicollinearity 
# 
# 3.What is Multiple Regression? 
# 
# b. A supervised learning algorithm used for regression problems involving multiple 
# independent variables. 
# 
# 4. Which of the following is an advantage of Multiple Regression over Simple Linear 
# Regression? 
# 
# c. Easier to implement 
# 
# 5. What is Polynomial Regression? 
#  
# b. A supervised learning algorithm used for regression problems that model the 
# relationship between the response variable and the independent variable as an nth 
# degree polynomial.
# 
# 6.Which of the following is NOT an assumption of Polynomial Regression? 
# 
# d. Independence 
# 
# 7.What is the coefficient of determination (R-squared) used for in Linear Regression? 
# 
# d. To measure the goodness of fit of the model. 
# 
# 8.Which of the following statements is true about Multicollinearity in Multiple Regression? 
# 
# b. It can lead to unstable estimates of the regression coefficients
# 
# 9.Which of the following statements is true about Overfitting in Polynomial Regression? 
# 
# b. It occurs when the model is too complex and fits the noise in the data. 
#  
# 10.Which of the following statements is true about Regularization in Linear Regression?
# 
# b. It is used to reduce the variance of the model. 
# 
# 11. Which of the following is an example of Linear Regression? 
# . 
# c. Predicting the price of a house based on its size and location.
# 
# 
# 
# # MCQs(more than one option could be correct): 
# 
# 1. A car rental company wants to predict the rental price of its cars based on the age of the 
# car and the number of miles driven. Which type of regression would be most appropriate 
# for this problem?
# 
# b. Multiple Regression 
#  
# 2. A clothing retailer wants to predict the sales of its products based on the price of the 
# product and the marketing spend on the product. However, the retailer suspects that 
# there might be a non-linear relationship between the price and the sales. Which type of 
# regression would be most appropriate for this problem? 
# 
# c. Polynomial Regressio
#  
# 3. A healthcare provider wants to predict the length of hospital stay for patients based on 
# their age, gender, medical history, and the severity of their illness. However, the provider 
# suspects that there might be a strong correlation between some of the independent 
# variables. Which technique can be used to address this issue? 
# 
# b. Regularization 
# 
# 4. A real estate agent wants to predict the selling price of a house based on its location, 
# size, number of bedrooms, and age. However, the agent suspects that the relationship 
# between the independent variables and the dependent variable might not be linear. 
# Which type of regression would be most appropriate for this problem? 
# 
# c. Polynomial Regression 
# 
# 5. A marketing agency wants to predict the conversion rate of a digital advertising 
# campaign based on the target audience, the ad creative, and the ad spend. However, 
# the agency suspects that there might be interactions between the independent variables. 
# Which technique can be used to address this issue?
# 
# d. Interaction Terms 
# 

# # Case Study: 
# 
# Predicting House Prices using Multiple Polynomial Regression In this case study, you will build a multiple polynomial regression model to predict the prices of houses based on their characteristics. You will use the "California Housing" dataset from sklearn, which contains information about houses in California.
# 
# # Dataset Description :
# 
# The dataset contains the following columns: ● MedInc: Median income of the block. ● HouseAge: Median age of the houses in the block. ● AveRooms: Average number of rooms per household in the block. ● AveBedrms: Average number of bedrooms per household in the block. ● Population: Number of people residing in the block. ● AveOccup: Average household occupancy in the block. ● Latitude: Latitude of the block in decimal degrees. ● Longitude: Longitude of the block in decimal degrees. ● Target: Median house value of the block in units of 100,000.

# # Importing the required libarary

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # The Data
# Reading the Data file from the sourse

# In[4]:


df=pd.read_csv(r"D:\data science\Assignment_19_april\housing.csv")


# In[5]:


df


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


print(df.corr())


# In[10]:


df.columns


# In[11]:


df.isnull().sum()


# In[23]:


#Remove rows with missing values
df1=df.dropna()
df1


# In[22]:


#Drop Latitude and Longitude columns
df.drop(labels=['longitude', 'latitude'], axis=1)


# #Visualize the dataset, histograms of all the features.

# In[27]:


plt.hist(df['housing_median_age'])

plt.show()


# In[28]:


plt.hist(df['total_rooms'])

plt.show()


# In[29]:


plt.hist(df['total_bedrooms'])

plt.show()


# In[30]:


plt.hist(df['population'])

plt.show()


# In[31]:


plt.hist(df['households'])

plt.show()


# In[32]:


plt.hist(df['households'])

plt.show()


# In[33]:


plt.hist(df['median_income'])

plt.show()


# In[34]:


plt.hist(df['median_house_value'])

plt.show()


# In[35]:


plt.hist(df['ocean_proximity'])

plt.show()


# In[49]:


from sklearn.model_selection import train_test_split


# In[69]:


#splitting
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)                                                         


# In[ ]:




