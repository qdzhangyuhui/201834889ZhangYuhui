# coding=utf-8
import os
import nltk
import re
import string
import math
import pandas as pd

rootpath = '..\\data'
wordlist = []  # 所有单词表
vectors = []  # 生产的文档向量集
docs = []  # 存储处理好的文档内容集合
dict=[]# 词典表
threshold_value_low=30
threshold_value_high=400
idflist={}
countlist={}

def doc_word_TF(doc):
    print('tf')
    doctf = {}
    for word in doc:
        if word not in doctf:
            doctf[word] = 1
        else:
            doctf[word] += 1
    return doctf


def word_IDF(word):
    count = 0
    print('idf df')
    for doc in docs:
        if word in set(doc):
            count = count + 1
    idf=math.log(len(docs) / count)
    idflist[word]=idf
    return math.log(len(docs) / count), count


def creatvectors():
  with open('..\\outdata\\vector0.csv', 'w') as fi:
    for doc in docs:
        vector = [0.0 for i in range(len(dict))]
        tflist = doc_word_TF(doc[0:-1])
        for word in doc[0:-1]:
          if word in dict:
            tf = tflist[word]
            #df=countlist[word]
            idf=idflist[word]
            #print(word,'tf=',tf)
            vector[dict.index(word)] = tf * idf
        vector.append(str(doc[-1]))
        strvector=str(vector).replace('[','')
        fi.write(strvector.replace(']',''))
        fi.write('\n')
        print(str(vector)+'\n')
        #vectors.append(vector)
  fi.close()



def main():
    # 遍历文件夹进行预处理
    label = 1
    print("data Processing")
    for folderlists in os.listdir(rootpath):
        # print(folderlists)
        path = os.path.join(rootpath, folderlists)
        # print(path)
        for file in os.listdir(path):
            print('current file:'+file)
            doc = []  # 存储处理好的文档
            filepath = os.path.join(path, file)
            if os.path.isfile(filepath):  # 是文件的话读取文件内容
                with open(filepath, mode='r', encoding='latin-1', errors="ignore") as f:
                    document = f.read()
                f.close()
                # 文件内容处理
                document = document.lower()  # 大写转小写
                tokens = nltk.word_tokenize(document)  # tokenizaton
                punctuation_remove = re.compile('[%s]' % re.escape(string.punctuation))  # 去标点符号
                tokens = list(filter(lambda word: word != "", [punctuation_remove.sub("", word) for word in tokens]))
                # print(tokens)
                stopwords = set(nltk.corpus.stopwords.words("english"))  # 停用词
                stem = nltk.stem.LancasterStemmer()  # 提取词干
                # print(tokens)
                for token in tokens:
                  if not token.isdigit()and token.isalpha():#去除数字
                    token = stem.stem(token)  # Stemming
                    if token not in stopwords:  # 去停用词
                        doc.append(token)#存储处理好的文章
                        if token not in wordlist:  # 去重
                            wordlist.append(token)
                        if token not in countlist :
                            countlist[token]=1
                        else  :
                            countlist[token]+=1
                doc.append(label)#为文章打标签
                docs.append(doc)
        label += 1#一个文件夹为一个标签
    # print(docs)
    # 过滤并输出输出维度表
    print('output wordlist')
    for word in wordlist:#过滤的word
        cf = countlist[word]
        print(word,'cf=',cf)
        if cf > threshold_value_low and cf < threshold_value_high:
            #print(word,df)
            idf,df=word_IDF(word)
            print(word, 'idf=', idflist[word])
            #countlist[word]=df
            dict.append(word)
    with open('..\\outdata\\wordlist0.txt', 'w', errors="ignore") as f:
        for w in dict:
            f.write(w + '\n')
    f.close()
    print('output wordlist finish')
    print('begin to create vectors')
    creatvectors()


if __name__ == '__main__':
    main()

