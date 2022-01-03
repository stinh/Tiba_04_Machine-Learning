
#!pip install opencc-python-reimplemented

from opencc import OpenCC
cc = OpenCC('s2tw') # 簡轉繁
to_convert = '开放中文转换' #轉換的內容
converted = cc.convert(to_convert) #進行轉換
converted #印出轉換結果

"""**利用Jieba做中文分詞**"""

# 用with as 寫法會自動關檔，才不會耗資源
with open("news.txt", "r", encoding="utf-8") as f:  
    article=f.read() # 讀檔
article # 列出文章

"""只有載入大辭典：分詞結果有誤差"""

import jieba
from urllib.request import urlretrieve #將URL表示的網路物件複製到本地檔案
url = "https://github.com/fxsjy/jieba/raw/master/extra_dict/dict.txt.big" # 載入大辭典，加強結巴中文分詞準確度
urlretrieve(url, "dict.txt.big")
jieba.set_dictionary("dict.txt.big")

" ".join(jieba.cut(article)) #把文章分詞後，分詞間加上空格後呈現

"""加載入自訂的辭典：分詞結果更精準"""

import jieba
from urllib.request import urlretrieve
# 1. 載入大辭典
url = "https://github.com/fxsjy/jieba/raw/master/extra_dict/dict.txt.big"
urlretrieve(url, "dict.txt.big")
jieba.set_dictionary("dict.txt.big")
# 2. 載入自定義辭典：根據文章自訂辭典後上傳到雲端取連結
url = "https://github.com/stinh/Tiba_04_Machine-Learning/raw/main/politics.txt"
urlretrieve(url, "politics.txt")
jieba.load_userdict("politics.txt")

" ".join(jieba.cut(article)) #把文章分詞後，分詞間加上空格後呈現

import jieba.analyse
# jieba.analyse.extract_tags(article) #進行分詞
# jieba.analyse.extract_tags(article, topK=None) #分詞數預設為20組，如不限定數要用None
jieba.analyse.extract_tags(article, topK=None, withWeight=True) # withWeight：該分詞在文章內重要性
# jieba.analyse.extract_tags(article, topK=None, allowPOS=["n", "nr", "ns"]) # allowPOS：挑出指定詞性的分詞