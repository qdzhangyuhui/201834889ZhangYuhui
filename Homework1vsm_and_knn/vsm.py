# coding=utf-8
import os
import nltk
import re
import string

rootpath = '..\\datatest'
wordlist = []



def main():
    # 遍历文件夹进行处理
    for folderlists in os.listdir(rootpath):
        # print(folderlists)
        path = os.path.join(rootpath, folderlists)
        # print(path)
        for file in os.listdir(path):
            # print(file)
            filepath = os.path.join(path, file)
            if os.path.isfile(filepath):#是文件的话读取文件内容
                with open(filepath, mode='r', encoding='latin-1') as f:
                    document = f.read()
                f.close()
                #文件内容处理
                document = document.lower()  # 大写转小写
                tokens = nltk.word_tokenize(document)  # tokenizaton
                punctuation_remove = re.compile('[%s]' % re.escape(string.punctuation))  # 去标点符号
                tokens = list(filter(lambda word: word != "", [punctuation_remove.sub("", word) for word in tokens]))
                #print(tokens)
                stopwords = set(nltk.corpus.stopwords.words("english"))#停用词
                stem = nltk.stem.LancasterStemmer()  # 提取词干
                # print(tokens)
                for token in tokens:
                    token = stem.stem(token)#Stemming
                    if token not in wordlist and token not in stopwords:#去重复词和停用词
                        wordlist.append(token)
    with open('word.txt', 'w') as f:
        for w in wordlist:
            f.write(w + '\n')
    f.close()


if __name__ == '__main__':
    main()
