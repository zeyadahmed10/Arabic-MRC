#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gdown
import pandas as pd
import json
from sklearn.model_selection import train_test_split


# In[2]:


from Preprocess.arabertpreprocess import ArabertPreprocessor


# In[3]:


url = 'https://drive.google.com/uc?id=1-3aZrNqg_TwX2WbPP7-BmDR1HwCMsdiL'
output = 'Asquadv2-train.csv'
gdown.download(url, output, quiet=False)


# In[3]:


df = pd.read_csv('Asquadv2-train.csv')


# In[4]:


print(df.shape)
print(df.columns)


# In[5]:


filt = df['answer_start']!=-1
cleaned_span_df = df[filt]
print(cleaned_span_df.shape)


# In[9]:


cleaned_span_df = cleaned_span_df.dropna(subset = ['title', 'context'])


# In[10]:


def dataframe2dict(df):
    model_name = "araelectra-base-discriminator"
    arabert_prep = ArabertPreprocessor(model_name=model_name)
    generated_data = dict()
    generated_data['version'] =2.0
    generated_data['data'] = list()
    df_title_g = df.groupby(['title'])
    title_keys = list(df_title_g.groups.keys())
    cnt = 0
    for key1 in title_keys:# first level
        new_df = df_title_g.get_group(key1)
        df_context_g = new_df.groupby(by=['context'])
        context_keys = list(df_context_g.groups.keys())
        key1_processed = arabert_prep.preprocess(key1)
        generated_data['data'].append({'title':key1_processed,'paragraphs':list()})
        for key2 in context_keys:#second level
            new_df_2 = df_context_g.get_group(key2)
            key2_processed = arabert_prep.preprocess(key2)
            triplet_dict = {'context':key2_processed, 'qas':list()}
            for idx, row in new_df_2.iterrows():
                qa_dict = {'question':arabert_prep.preprocess(row['question']), 'id':row['ID'], 'is_impossible':row['is_impossible']}
                if row['is_impossible'] ==True:
                    qa_dict['answers'] = [{'text':arabert_prep.preprocess(row['answer']), 'answer_start':row['answer_start']}]
                else:
                    qa_dict['plausible_answers'] =[{'text':arabert_prep.preprocess(row['answer']), 'answer_start':row['answer_start']}]
                    qa_dict['answers']=list()
                triplet_dict['qas'].append(qa_dict)
                cnt = cnt+1
            generated_data['data'][-1]['paragraphs'].append(triplet_dict)
    return generated_data


# In[11]:


df_train, df_test, y_train, y_test = train_test_split(cleaned_span_df, cleaned_span_df['is_impossible'],test_size = 0.1,stratify = cleaned_span_df['is_impossible'])


# In[12]:


# 0.111111 x 0.9 = 0.1
df_train, df_val, y_train, y_val = train_test_split(df_train, df_train['is_impossible'],test_size = 0.111111,stratify = df_train['is_impossible'])


# In[13]:


print(df_train.shape, df_val.shape, df_test.shape)


# In[18]:


train_dataset = dataframe2dict(df_train)
val_dataset = dataframe2dict(df_val)
test_dataset = dataframe2dict(df_test)


# In[19]:


with open("Data/asquadv2-train.json", "w") as outfile:
    json.dump(train_dataset, outfile)
with open("Data/asquadv2-val.json", "w") as outfile:
    json.dump(val_dataset, outfile)
with open("Data/asquadv2-test.json", "w") as outfile:
    json.dump(test_dataset, outfile)

