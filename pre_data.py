
# coding: utf-8

import json as js
import pandas
from collections import defaultdict

def OpenData():
    with open("E://1数据挖掘/C题全部数据/train_data_complete.json",'r', encoding='UTF-8') as load_f:
        train_data_sample = js.load(load_f)
        #print(train_data_sample[4])
        
    with open("E://1数据挖掘/C题全部数据/test_data_sample.json",'r', encoding='UTF-8') as load_f:
        test_data_sample = js.load(load_f)
        #print(test_data_sample[4])
    return train_data_sample,test_data_sample

train_data_sample,test_data_sample = OpenData()


train_data_sample[0]



#提取出问题答案的文字

A,L,item_id,Q = [],[],[],[]                 #储存为 问题 答案 标签
#print(len(load_dict))
for i in range(0,27000):
    for j in range(0,len(train_data_sample[i]['passages'])):
        A.append(train_data_sample[i]['passages'][j]['content'])                     #答案
        #print(A)
        L.append(train_data_sample[i]['passages'][j]['label'])                       #标签
        item_id.append(train_data_sample[i]['item_id'])        #id
        Q.append(train_data_sample[i]['question'])
      

    

A2,L2,item_id2,Q2 = [],[],[],[]                 #储存为 问题 答案 标签
#print(len(load_dict))
for i in range(27000,len(train_data_sample)):
    for j in range(0,len(train_data_sample[i]['passages'])):
        A2.append(train_data_sample[i]['passages'][j]['content'])                     #答案
        #print(A)
        L2.append(train_data_sample[i]['passages'][j]['label'])                       #标签
        item_id2.append(train_data_sample[i]['item_id'])        #id
        Q2.append(train_data_sample[i]['question'])

c0,c1 = 0,0
for i in L:
    if i==1:
        c0+=1
    else:
        c1+=1
for i in L2:
    if i==1:
        c0+=1
    else:
        c1+=1
print(c0,c1)


print(len(A)+len(A2))


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


import jieba
addstopwords()


list(jieba.cut(Q[0].strip(),cut_all=False))

list(jieba.cut('天空为什么是蓝色的'.strip(),cut_all=False))


'_'.join(jieba.cut('世界上有多少个国家?'.strip(),cut_all=False))

import time
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
max_len = 0
sum_len = 0
for i in range(0,len(A)):
    c_len = len(list(jieba.cut(A[i].strip(),cut_all=False)))
    if c_len>max_len:
        max_len = c_len
        #print('更新长度')
    else:
        pass
    sum_len+=c_len
mean_len = sum_len/len(A)        
print('最大长度为',max_len)
print('平均长度为',mean_len)
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

def padding(line):
    _len = 100
    if len(line)<_len:
        for pa in range(_len-len(line)):
            line.append('<a>')
    else:
        line = line[0:100]
        pass
    return line
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  

# 对句子进行分词  
def rm_tokens(sentence): 
    stopwords = stopwordslist('stop_words.txt')  
    output = []
    for word in sentence:  
        if word not in stopwords:  
            if word != '\t':  
                output.append(word)
    return output


stopwords = stopwordslist('stop_words.txt')
print(''.join(stopwords))

print(rm_tokens(jieba.cut('世界上•有 几个国家',cut_all=False)))
'_'.join(padding(list(rm_tokens(jieba.cut('世界上       有•几个国家'.strip().replace(' ',''),cut_all=False)))))

print(len(A),len(Q),len(A2),len(Q2))


import time
train_list = []
test_list = []
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#for i in range(0,5):
for i in range(0,len(A)):
    tq = '_'.join(padding(list(rm_tokens(jieba.cut(Q[i].strip().replace(' ',''),cut_all=False)))))
    ta = '_'.join(padding(list(rm_tokens(jieba.cut(A[i].strip().replace(' ',''),cut_all=False)))))
    tl = 'qid:'+str(item_id[i])
    train_list.append(' '.join([str(L[i]),tl,tq,ta]))
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#for i in range(0,5):
for i in range(0,len(A2)):
    tq2 = '_'.join(padding(list(rm_tokens(jieba.cut(Q2[i].strip().replace(' ',''),cut_all=False)))))
    ta2 = '_'.join(padding(list(rm_tokens(jieba.cut(A2[i].strip().replace(' ',''),cut_all=False)))))
    tl2 = 'qid:'+str(item_id2[i])
    test_list.append(' '.join([str(L[i]),tl2,tq2,ta2]))
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

ntrain_list = '\n'.join(train_list)

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

ntest_list = '\n'.join(test_list)

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

import re
f = open("D://1数据挖掘/C_pre_data/train", "w", encoding='utf-8')
f.write(ntrain_list)
f.close()

f = open("D://1数据挖掘/C_pre_data/test", "w", encoding='utf-8')
f.write(ntest_list)
f.close()

print('训练/测试数据写入成功')

