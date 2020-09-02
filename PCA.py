#!/usr/bin/env python
# coding: utf-8

# In[22]:


#here we import all the require packages
import numpy as np
import pandas as pd


# In[49]:


df =pd.read_csv ('Desktop/PCA_practice_dataset.csv')#load the data set
df.head()


# In[18]:


#convert the data into numpy
x=df.to_numpy()
x.shape


# In[19]:


#we need to scale the data with mean 0 and SD=1 before we apply PCA
from sklearn.preprocessing import StandardScaler


# In[42]:


scaler=StandardScaler()
df_std=scaler.fit_transform(df)
scaler.fit(df)


# In[44]:


#calculatingcovariance matrix
cov_mat = np.cov(df_std.T)
print(cov_mat)


# In[48]:


#calculating eygon values and eygon vectors
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print(eigen_vals)
print(eigen_vecs)


# In[23]:


#Applying Principal Compoet analysis
from sklearn.decomposition import PCA


# In[26]:


pca=PCA()
X=pca.fit_transform(x)


# In[27]:


#get a sum of all the eigon values fo corrosponding eigon vectors
c_varience = np.cumsum(pca.explained_variance_ratio_)*100
print(c_varience )


# In[28]:


#we declared list of our desire thresholds
thresholds=[i for i in range(90,97+1,1)]
print(thresholds )


# In[37]:


#here we make a list of compoents required to retain the variace as mentioned in our required threshold list
components =[np.argmax(c_varience>threshold)for threshold in thresholds]
print(components)
for component, threshold in zip(components,thresholds):
    print("Components required for{}% threshold are:{}".format(threshold,component))


# In[38]:


import matplotlib.pyplot as plt
plt.plot(components,range(90,97+1,1),'ro-',linewidth=2)
plt.title('Scree Plot')
plt.xlabel('principal component')
plt.ylabel('Threshold in %')
plt.show()


# In[40]:


#now we perform dimensionality reduction
X_orig=X
for component,var in zip(components,thresholds):
    pca=PCA(n_components=component)
    X_transformed=pca.fit_transform(X_orig)
    print('performing d.r.to retain {}% threshold'.format(var))
    print('after performing d.r. ,new shape of the dataset is:',X_transformed.shape)
    print('\n')


# In[ ]:




