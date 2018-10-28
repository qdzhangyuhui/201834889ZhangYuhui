# coding=utf-8
import os
import nltk
import re
import string
rootpath = '..\\datatest'
#遍历文件夹
for folderlists in os.listdir(rootpath):
    print(folderlists)
    path = os.path.join(rootpath, folderlists)
    print(path)
    for file in os.listdir(path):
        print(file)
        filepath = os.path.join(path,file)
        if os.path.isfile(filepath):
           with open(filepath, mode='r',encoding='latin-1') as f:
               document = f.read()
           f.close()
           #tokenization
           tokens = nltk.word_tokenize(document)
           print(tokens)
           #print(document)

def main():
   print()

if __name__ == '__main__':
    main()