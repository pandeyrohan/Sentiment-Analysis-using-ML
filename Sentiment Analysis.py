#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests


# In[4]:


from bs4 import BeautifulSoup


# In[6]:


r = requests.get('https://www.yelp.com/biz/tesla-san-francisco?osq=Tesla+Dealership')


# In[10]:


print(r.status_code)


# In[11]:


r.text


# In[29]:


# Make the soup
soup = BeautifulSoup(r.text, 'html.parser')


# In[38]:


# First get all of the review-content divs
results = soup.findAll(class_='comment__09f24__gu0rG css-qgunke')
print(results)


# In[41]:


# Loop through review-content divs and extract paragraph text
reviews = []
for result in results:
  reviews.append(result.find('span').text)


# In[44]:


for review in reviews:
    print(review,'\n')


# In[46]:


reviews[0]


# In[47]:


#ANALYSE THE DATA


# In[49]:


import numpy as np
import pandas as pd


# In[53]:


# Create a pandas dataframe from array
df = pd.DataFrame(np.array(reviews), columns=['review'])


# In[55]:


df.head()


# In[57]:


len(df['review'])


# In[59]:


df['word_count'] = df['review'].apply(lambda x: len(str(x).split(" ")))


# In[61]:


# Calculate character count
df['char_count'] = df['review'].str.len()


# In[63]:


def avg_word(review):
  words = review.split()
  return (sum(len(word) for word in words) / len(words))

# Calculate average words
df['avg_word'] = df['review'].apply(lambda x: avg_word(x))


# In[66]:


get_ipython().system('pip install nltk')


# In[73]:


# Import stopwords
import nltk


# In[75]:


nltk.download('stopwords')


# In[77]:


from nltk.corpus import stopwords


# In[79]:


stop_words = stopwords.words('english')
df['stopword_coun'] = df['review'].apply(lambda x: len([x for x in x.split() if x in stop_words]))


# In[81]:


df.describe()


# In[82]:


df.head()


# In[84]:


# Lower case all words
df['review_lower'] = df['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# In[85]:


# Remove Punctuation
df['review_nopunc'] = df['review_lower'].str.replace('[^\w\s]', '')


# In[87]:


stop_words = stopwords.words('english')


# In[89]:


# Remove Stopwords
df['review_nopunc_nostop'] = df['review_nopunc'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))


# In[91]:


# Return frequency of values
freq= pd.Series(" ".join(df['review_nopunc_nostop']).split()).value_counts()[:30]
print(freq)


# In[93]:


other_stopwords = ['tesla', 'get', 'service', 'tires', 'car', '4', 'back', 'went'   'said', 'model', 'came', 'time', 'cars', 'new', 'one',   'tire', 'us', 'extremely', 'call', 'front', 'also', 'today', 'insurance',   'department', 'whole', 'first', 'another', 'center','month','nothing']


# In[94]:


df['review_nopunc_nostop_nocommon'] = df['review_nopunc_nostop'].apply(lambda x: "".join(" ".join(x for x in x.split() if x not in other_stopwords)))


# In[96]:


df.head()


# In[98]:


get_ipython().system(' pip install textblob')


# In[101]:



from textblob import Word


# In[103]:


# Lemmatize final review format
nltk.download('wordnet')
df['cleaned_review'] = df['review_nopunc_nostop_nocommon'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# In[104]:


# Calculate polarity
from textblob import TextBlob


# In[105]:


df['polarity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment[0])


# In[107]:


df[['review','polarity']]


# In[109]:


# Calculate subjectivity
df['subjectivity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment[1])


# In[110]:


df[['review','subjectivity']]


# In[ ]:




