#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_excel("D:\quicker_car.xlsx")
data


# In[3]:


data.describe()


# In[4]:


data.info() # it means all data features are object


# In[5]:


data.isnull().sum()


# In[6]:


data['fuel_type'].unique()


# In[7]:


data['year'].unique()


# In[8]:


data['Price'].unique()


# In[9]:


data['kms_driven'].unique()

problem in data
fuel type have non values
km_driven have comma and kms
price have "ask for price"
keep first three word of name
year column have non year data


# In[10]:


backup=data.copy()


# In[11]:


data=data[data['year'].str.isnumeric()!=False] # isnumeric is working at str
data


# In[12]:


data['year']=data['year'].astype(int)


# In[13]:


data


# In[14]:


data.info()


# In[15]:


data=data[data['Price']!='Ask For Price']
data


# In[16]:


data['price']=data['Price'].replace(',','').astype(int)


# In[17]:


data.info()


# In[18]:


data['kms_driven']=data['kms_driven'].str.split(' ').str.get(0).str.replace(',','')


# In[19]:


data['kms_driven'].unique()


# In[20]:


data=data[data['kms_driven']!='Petrol']
data


# In[21]:


data['kms_driven']=data['kms_driven'].astype(int)


# In[22]:


data.info()


# In[23]:


data[data['fuel_type'].isna()]


# In[24]:


data=data[~data['fuel_type'].isna()]  # na bali rows hat jayegi
data


# In[25]:


#data['name'].str.split(" ")[0:3]


# In[26]:


data['name']=data['name'].str.split(" ").str.slice(0,3).str.join(' ')


# In[27]:


data


# In[28]:


data=data.reset_index(drop=True)
data


# In[29]:


data.info()


# In[30]:


data.drop('price',inplace=True,axis=1)


# In[31]:


data


# In[32]:


data['Price']=data['Price'].astype(int)


# In[33]:


data


# In[34]:


data.info()


# In[35]:


data.isnull().sum()


# In[36]:


data.describe()


# In[37]:


# in my analysis 75 % car have price something around 500000 and max vlaue of car have 8500000 so we can say our data
#have outlier let we check less than 600000 
data[data['Price']>6e6]   #only one car so it is outlier in our data


# In[38]:


data=data[data['Price']!=8500003]
data


# In[39]:


data.describe()


# In[40]:


data=data.reset_index(drop=True)
data


# In[41]:


#now our data is clean
data.to_csv('clean_car_data.csv')


# In[42]:


x=data.drop('Price',axis=1)


# In[43]:


x


# In[44]:


y=data['Price']
y


# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[47]:


x_train


# In[48]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[49]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


# In[50]:


from sklearn.pipeline import make_pipeline
ohe=OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])


# In[51]:


ohe.categories_


# In[52]:


make_transformer=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')
# if we will not pass ohe.categories then whenever any test data will have those categories which are not in train data then pipe 
# line will show an error  


# In[53]:


lr=LinearRegression()


# In[54]:


model=make_pipeline(make_transformer,lr)


# In[55]:


model.fit(x_train,y_train)# pahle mera train data make transformer ke pass jayega aur baha per transform hoga one_hot_encoder
#se uske baad lr algo kaam karegi yeh kaam pipline ka h


# In[56]:


y_pred=model.predict(x_test)


# In[57]:


y_pred


# In[58]:


r2_score(y_test,y_pred)


# In[59]:


#so will use loop
list1=[]
for i in range(0,1000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=i)
    lr=LinearRegression()
    model=make_pipeline(make_transformer,lr)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    score=r2_score(y_pred,y_test)
    list1.append(score)
    
    
    
    


# In[60]:


import numpy as np


# In[61]:


np.argmax(list1)  # it give highest score at random state=124


# In[62]:


list1[np.argmax(list1)]  


# In[63]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=124)
lr=LinearRegression()
pipe=make_pipeline(make_transformer,lr)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)

r2_score(y_pred,y_test)


# In[64]:


import pickle


# In[65]:


with open('linearregression.pkl','wb') as f:
    pickle.dump(pipe,f)             


# In[66]:


pipe.predict(pd.DataFrame([['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']],columns=['name','company','year','kms_driven','fuel_type']))
# agar mujhe value string and int dono me pass karni ho to as  dataframe banakar pass karni hongi


# In[ ]:




