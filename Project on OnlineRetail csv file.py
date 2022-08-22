#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("OnlineRetail.csv")


# In[3]:


data


# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


data.head(387961)


# In[8]:


data.dtypes


# In[9]:


data.isna().sum()


# In[10]:


data.describe()


# In[11]:


data['InvoiceNo'] = np.where(data['InvoiceNo'] == 'yes',1,0)
data['IInvoiceNo'] = data['InvoiceNo']
type(data["InvoiceNo"])


# In[12]:


import numpy as np
data['CustomerID'] = np.where(data['CustomerID'] == 'yes',1,0)
data['CustomerID'] = data['CustomerID'].astype(str)


# In[13]:


data.isna().sum()


# In[14]:


print(data['StockCode'].mode())


# In[15]:


data.Description


# In[16]:


print(data['Quantity'].mode())


# In[17]:


int(data['Quantity'].median())


# In[18]:


data.columns


# In[19]:


col = list (data.columns)


# In[20]:


col


# In[21]:


import matplotlib.pyplot as plt


# In[22]:


data.hist()


# In[23]:


import seaborn as sns
sns.boxplot()


# In[24]:


data.describe()


# In[25]:


sns.boxplot(x=data['UnitPrice'])


# In[26]:


sns.boxplot(x=data['Quantity'])


# In[27]:


sns.boxplot(x = data['InvoiceNo'])


# In[28]:


data.hist()


# In[29]:


data.describe()


# In[30]:


Q1A = data.UnitPrice.quantile(0.25)
Q3A = data.Quantity.quantile(0.75)
IQRA= Q3A-Q1A
print(IQRA)


# In[31]:


print(Q1A-1.5*IQRA)
print(Q3A+1.5*IQRA)


# In[32]:


Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[33]:


data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[34]:


data.head(10)


# In[35]:


data.shape


# In[36]:


col_list = list(data.columns)


# In[37]:


col_list


# In[38]:


col_list.remove('InvoiceNo')


# In[39]:


col_list


# In[40]:


for col in col_list:
    if data[col].dtypes=='object':
        #print('ob')
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        #print('num')
        data[col] = data[col].fillna(data[col].mean())


# In[41]:


data.isna().sum()


# In[42]:


for col in data.columns :
         if ((data[col].dtype == 'object') & (col != 'StockCode') ):
             col_list.append(col)


# In[43]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for i in col_list:
    data[i]=labelencoder.fit_transform(data[i])
    dfmi = pd.DataFrame([list('data'),
                       list('efgh'),
                       list('ijkl'),
                       list('mnop')],
                     columns=pd.MultiIndex.from_product([['one', 'two'],
                                                         ['first', 'second']]))
   


# In[44]:


data


# In[45]:


pd.options.display.float_format = '{:,.2f}'.format


# In[46]:


display


# In[47]:


data.dtypes


# In[48]:


col_list = []
for col in data.columns:
    if ((data[col].dtype == 'object') & (col != 'UnitPrice') ):
        col_list.append(col)


# In[49]:


col_list


# In[50]:


col_list
X=data[col_list]
X


# In[51]:


data


# In[54]:


data


# In[55]:


data.hist()


# In[56]:


data.hist('StockCode')


# In[57]:


data.hist('Description')


# In[58]:


data.hist('Country')


# In[59]:


data.hist('InvoiceDate')


# In[60]:


x = pd.DataFrame(data['Country'])


# In[61]:


x


# In[62]:


y = pd.DataFrame(data['Country'])


# In[63]:


y


# In[64]:


from sklearn.linear_model import LinearRegression


# In[65]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state= 5)


# In[66]:


from sklearn.linear_model import LinearRegression


# In[67]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state= 5)


# In[68]:


lin_fit=LinearRegression()


# In[69]:


from sklearn.model_selection import train_test_split


# In[70]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)


# In[71]:


from sklearn.model_selection import train_test_split


# In[72]:


from sklearn.linear_model import LinearRegression


# In[73]:


lin_reg=LinearRegression().fit(x_train,y_train)
lin_reg


# In[74]:


x_pred=lin_reg.predict(y_test)


# In[75]:


x_pred


# In[76]:


y_pred=lin_reg.predict(x_test)


# In[77]:


y_pred


# In[ ]:




