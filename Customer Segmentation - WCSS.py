#!/usr/bin/env python
# coding: utf-8

# # Importing the dependencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# # Data Collection and pre-processing

# In[2]:


cus_data = pd.read_csv("D:\ML Files\Customer Segmentation by K-means\mall_customers.csv")


# In[3]:


cus_data.head()


# In[4]:


cus_data.shape


# In[5]:


cus_data.info()


# In[6]:


#Missing value
cus_data.isnull().sum()


# Selecting features and labels

# In[12]:


X = cus_data.iloc[:,[3,4]].values
print(X)


# # Choosing the number of clusters

# In[14]:


#WCSS - Within CLusters Sum of Squares
#Finding wcss for diff clusters
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)


# In[23]:


#Plot
sns.set()
plt.plot(range(1,11),wcss)
plt.title("The Elbow point graph")
plt.xlabel("No of Clusters")
plt.ylabel("WCSS")
plt.show()


# Optimum no of clusters = 5

# In[24]:


kmeans = KMeans(n_clusters=5,init="k-means++",random_state=42)

#Return a label for each datapoint based on their clusters
Y = kmeans.fit_predict(X)


# In[29]:


print(Y)


# In[34]:


#Visualize all the clusters
plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0],X[Y==0,1],s=50,c='green',label = "Cluster 1")
plt.scatter(X[Y==1,0],X[Y==1,1],s=50,c='red',label = "Cluster 2")
plt.scatter(X[Y==2,0],X[Y==2,1],s=50,c='blue',label = "Cluster 3")
plt.scatter(X[Y==3,0],X[Y==3,1],s=50,c='yellow',label = "Cluster 4")
plt.scatter(X[Y==4,0],X[Y==4,1],s=50,c='brown',label = "Cluster 5")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='cyan',label="Centroids")
plt.title("Clusters")
plt.xlabel("Annual Income")
plt.ylabel("Spending Scores")
plt.show()


# In[ ]:





# In[ ]:




