
# coding: utf-8

# In[1]:


def addstopwords():
    #询问数字
    jieba.add_word('几')
    jieba.del_word('几年')
    jieba.del_word('几月')
    jieba.del_word('几日')
    jieba.del_word('几个')
    jieba.add_word('多少')
    #询问时间
    jieba.add_word('什么时候')
    jieba.add_word('何时')
    
    #询问原因
    jieba.add_word('为什么')
    jieba.add_word('为何')
    jieba.add_word('什么原因')
    #询问地方
    jieba.add_word('哪里')
    jieba.add_word('哪个')
    jieba.add_word('那里')
    #询问动作
    jieba.add_word('怎么')
    jieba.add_word('如何')
    jieba.add_word('怎样')
    #询问人
    jieba.add_word('谁')
def convert_doc_to_wordlist(str_doc,cut_all):
    # 分词的主要方法
    sent_list = str_doc.split('\n')
    sent_list = map(rm_char, sent_list) # 去掉一些字符，例如\u3000  map() 会根据提供的函数对指定序列做映射
    #print(sent_list)
    word_2dlist = [rm_tokens(jieba.cut(part,cut_all=cut_all)) for part in sent_list] # 分词
    #print(word_2dlist)
    word_list = sum(word_2dlist,[])
    return word_list

def rm_char(text):
    text = re.sub('\u3000','',text)
    return text

def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  

# 对句子进行分词  
def rm_tokens(sentence): 
    stopwords = stopwordslist('stop_words.txt')  # 这里加载停用词的路径  
    output = []
    for word in sentence:  
        if word not in stopwords:  
            if word != '\t':  
                output.append(word)  
                
    return output  


# In[2]:


import json as js
import pandas
from collections import defaultdict

def OpenData():
    with open("D://1数据挖掘/C题全部数据/train_data_complete.json",'r', encoding='UTF-8') as load_f:
        train_data_sample = js.load(load_f)
        #print(train_data_sample[4])

    with open("D://1数据挖掘/C题全部数据/test_data_sample.json",'r', encoding='UTF-8') as load_f:
        test_data_sample = js.load(load_f)
        #print(test_data_sample[4])
    return train_data_sample,test_data_sample


# In[3]:


train_data_sample,test_data_sample = OpenData()


# In[4]:


#提取出问题的文字
from pprint import pprint
Q = []
A,L,passage_id,Q2 = [],[],[],[]       #储存为 问题 答案 标签
#print(len(load_dict))
for i in range(0,len(train_data_sample)):
    #print(i)
    Q.append(train_data_sample[i]['question'])
    #print(Q)
    for j in range(0,len(train_data_sample[i]['passages'])):
        A.append(train_data_sample[i]['passages'][j]['content'])
        #print(A)
        L.append(train_data_sample[i]['passages'][j]['label'])
        passage_id.append(train_data_sample[i]['passages'][j]['passage_id'])
        


# In[12]:


#作映射
from gensim import corpora,models
import jieba
import re
from pprint import pprint
import time

print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
files = Q+A
dictionary = corpora.Dictionary()
for file in files:
    file = convert_doc_to_wordlist(file, cut_all=False)
    dictionary.add_documents([file])
#pprint(sorted(list(dictionary.items()),key=lambda x:x[0]))
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


# In[13]:


#去掉低频词
small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 5 ]
dictionary.filter_tokens(small_freq_ids)
dictionary.compactify()
#pprint(sorted(list(dictionary.items()),key=lambda x:x[0]))


# In[14]:


from gensim.models import word2vec
model = word2vec.Word2Vec.load(u'D:/1数据挖掘/C_pre_data/word2vec_model')


# In[15]:


len(dictionary)


# In[17]:


import numpy as np
import time
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
vector = []
#for i in range(100):
for i in range(len(dictionary)):
    try:
        #print(vv[i])
        v = ' '.join(str(x) for x in model[dictionary[i]])
        vector.append(dictionary[i]+' '+v)
    except:
        v = ' '.join(str(x) for x in np.zeros(200))
        vector.append(dictionary[i]+' '+v)
    if i%10000==0:
        print (i,'steps at',time.strftime("%H:%M:%S", time.localtime()))
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("finished!")


# In[20]:


vector = '\n'.join(vector)
fo = open("D://1数据挖掘/C_pre_data/vectors.nobin","w", encoding='UTF-8')
fo.write(vector)
print("finished!")
fo.close()

