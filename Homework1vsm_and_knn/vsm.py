# coding=utf-8
import os
import nltk
import re
import string

rootpath = '..\\datatest'
wordlist = []


# 遍历文件夹


def main():
    for folderlists in os.listdir(rootpath):
        # print(folderlists)
        path = os.path.join(rootpath, folderlists)
        # print(path)
        for file in os.listdir(path):
            # print(file)
            filepath = os.path.join(path, file)
            if os.path.isfile(filepath):
                with open(filepath, mode='r', encoding='latin-1') as f:
                    document = f.read()
                f.close()
                # tokenizaton and normalization
                document = document.lower()  # 大写转小写
                tokens = nltk.word_tokenize(document)  # tokenizaton
                punctuation_remove = re.compile('[%s]' % re.escape(string.punctuation))  # 去标点符号
                tokens = list(filter(lambda word: word != "", [punctuation_remove.sub("", word) for word in tokens]))
                print(tokens)
                stem = nltk.stem.LancasterStemmer()  # 提取词干
                # print(tokens)
                for token in tokens:
                    token = stem.stem(token)#Stemming
                    #print(token)
                    #wordlist.append(token)
                    if token not in wordlist:
                       wordlist.append(token)
    with open('word.txt', 'w') as f:
        for w in wordlist:
            f.write(w + '\n')
    f.close()


if __name__ == '__main__':
    main()
