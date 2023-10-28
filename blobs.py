#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd;
d1=pd.read_csv('cluster_blobs.csv');
print(d1.head(10));


# In[4]:


print(d1.describe());


# In[24]:


from sklearn.cluster import DBSCAN;
db1_blobs=DBSCAN(eps=4,min_samples=4);
db1_blobs.fit(d1);


# In[25]:


l1=db1_blobs.labels_;
print(len(l1));


# In[26]:


for i in l1:
     print(i);


# In[27]:


no_of_cluster1=len(set(l1))-(1 if -1 in l1 else 0);
print(no_of_cluster1);


# In[28]:


import matplotlib.pyplot as plt;
import numpy as np;
X=np.array(d1);
Y=db1_blobs.fit_predict(X);
plt.figure(figsize=(8,5));
plt.scatter(X[Y==0,0],X[Y==0,1],s=5,c='green');
plt.scatter(X[Y==1,0],X[Y==1,1],s=5,c='red');
plt.scatter(X[Y==2,0],X[Y==2,1],s=5,c='yellow');
plt.xlabel('A');
plt.ylabel('B');
plt.show();


# In[29]:


from sklearn import metrics
print(metrics.silhouette_score(X,l1));


# In[32]:


from sklearn.cluster import KMeans;
k1= d1;
print(k1.head(10));


# In[33]:


from sklearn.preprocessing import StandardScaler;
scaler = StandardScaler();
df_scaled= scaler.fit_transform(k1);
print(df_scaled);


# In[43]:


k_means=KMeans(n_clusters=3,random_state=250);
k_means.fit(df_scaled);
k1['Clusters'] = k_means.labels_;
print(k1.head(10));


# In[39]:


print(k1.Clusters.value_counts());


# In[41]:


import matplotlib.pyplot as plt;
import numpy as np;
X=np.array(d1);
Y=k1.Clusters;
plt.figure(figsize=(8,5));
plt.scatter(X[Y==0,0],X[Y==0,1],s=5,c='green');
plt.scatter(X[Y==1,0],X[Y==1,1],s=5,c='red');
plt.scatter(X[Y==2,0],X[Y==2,1],s=5,c='yellow');
plt.xlabel('A');
plt.ylabel('B');
plt.show();


# In[42]:


print(metrics.silhouette_score(X,k1['Clusters']));

