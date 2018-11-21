import os
rootpath = '..\data'
docs = []#存储处理好的文档
words = []#词汇表
dict = [] #词典表(过滤后）

def dataprocessing(rootpath):
    print('input data')
    for folderlists in os.listdir(rootpath):
        # print(folderlists)
        path = os.path.join(rootpath, folderlists)
        # print(path)
        for file in os.listdir(path):
            print('current file:' + file)
            doc = []  # 存储处理好的文档
            filepath = os.path.join(path, file)
            if os.path.isfile(filepath):  # 是文件的话读取文件内容
                with open(filepath, mode='r', encoding='latin-1', errors="ignore") as f:
                    document = f.read()
                f.close()
                # 文件内容处理
                document = document.lower()  # 大写转小写
def main():
    print('NBC')
if __name__ == '__main__':
    main()