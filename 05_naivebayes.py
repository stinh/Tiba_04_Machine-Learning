
import pandas as pd
from urllib.request import urlretrieve
# 先把訓練資料上傳到雲端，並載入資料，轉為繁中pd表格形式
url = "https://github.com/stinh/Tiba_04_Machine-Learningraw/main/poem_train.csv"
urlretrieve(url, "train.csv")
train_df = pd.read_csv("train.csv", encoding="utf-8")

# 先把驗證資料上傳到雲端，並載入資料，轉為繁中pd表格形式
url = "https://github.com/stinh/Tiba_04_Machine-Learning/raw/main/poem_test.csv"
urlretrieve(url, "test.csv")
test_df = pd.read_csv("test.csv", encoding="utf-8")

# train_df
test_df

u = train_df["作者"].unique()
name2idx = {n:i for i, n in enumerate(u)}
idx2name = {i:n for i, n in enumerate(u)}
y_train = train_df["作者"].replace(name2idx)
y_test = test_df["作者"].replace(name2idx)
y_test

# train_df # 確認訓練資料
# test_df # 確認驗證資料

"""pandas 的 apply 是一個在 pandas dataframe 加入新列（Column）的指令。"""

# Series.apply(func)
import jieba
# s = "百川日東流，客去亦不息。我生苦漂蕩，何時有終極。"
# " ".join(jieba.cut(s))
def poemcut(p):
    return " ".join(jieba.cut(p))
x_train = train_df["內容"].apply(poemcut)
x_test = test_df["內容"].apply(poemcut)
x_test

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
x_train_count = vec.fit_transform(x_train) #fit找出有幾個分類；transform計算各分類在每首詩出現次數
x_test_count = vec.transform(x_test) #驗證資料不做fit(套用訓練資料的)，直接transform計算各分類計數，如果驗證資料出現測試資料沒有的欄位，就捨去那個資料

'''
看每一個分詞的出現位置
vec.vocabulary_
'''

# check標點和換行沒被算進去(。, \n, \r\n): KeyError才是對的
# vec.vocabulary_["。"]
# vec.vocabulary_["\n"]
# vec.vocabulary_["\r\n"]
# vec.vocabulary_["深情"]

x_train

# print(x_train_count)
x_train_count
x_test_count

from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()  # alpha 保底次數預設為1
clf = MultinomialNB(alpha=0.5) # alpha值建議為實際數值的百分之一或千分之一，但還是要去try，找出最佳accuracy_score時的alpha值
clf.fit(x_train_count, y_train)

from sklearn.metrics import accuracy_score
pre = clf.predict(x_test_count)
accuracy_score(pre, y_test)

"""confusion_matrix 混淆矩陣是機器學習中總結分類模型預測結果的情形分析表，以矩陣形式將資料集中的記錄按照真實的類別與分類模型作出的分類判斷兩個標準進行彙總。官方文件中給出的用法是：sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)

* y_true: 是樣本真實分類結果，y_pred: 是樣本預測分類結果
* labels：是所給出的類別，通過這個可對類別進行選擇
* sample_weight : 樣本權重
"""

from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_test, pre),
             index=["{}(正確)".format(n) for n in u], 
             columns=["{}(預測)".format(n) for n in u])

testp = input("來首詩:")
testpcount = vec.transform([poemcut(testp)])
proba = clf.predict_proba(testpcount)[0]
for n, p in zip(u, proba): 
    print(n, "的機率是:", p)

list(zip([1, 2], ["a", "b"], ["c", "d"])) #zip 可以各list中將相同序位的拉出來組成新的