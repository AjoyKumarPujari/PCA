#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.datasets import load_breast_cancer


# In[6]:


cancer=load_breast_cancer()


# In[7]:


cancer.keys()


# In[8]:


print (cancer['DESCR'])


# In[10]:


df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[11]:


df.head(5)


# In[13]:


from sklearn.preprocessing import StandardScaler


# In[15]:


scaler=StandardScaler()
scaler.fit(df)


# In[16]:


scaler_data=scaler.transform(df)


# In[18]:


scaler_data


# In[21]:


from sklearn.decomposition import PCA


# In[22]:


pca=PCA(n_components=2)


# In[23]:


pca.fit(scaler_data)


# In[26]:


x_pca=pca.transform(scaler_data)


# In[28]:


scaler_data.shape


# In[29]:


x_pca.shape


# In[30]:


scaler_data


# In[31]:


x_pca


# In[34]:


plt.figure(figsize=(10,8))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('First Principal Component')
plt.ylabel('Second principal Componet')


# In[ ]:




