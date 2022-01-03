
from urllib.request import urlretrieve
url = "https://github.com/stinh/Tiba_04_Machine-Learning/raw/main/train.csv"
urlretrieve(url, "train.csv")

url = "https://github.com/stinh/Tiba_04_Machine-Learning/raw/main/test.csv"
urlretrieve(url, "test.csv")

"""**資料以表格呈現**"""

import pandas as pd
train = pd.read_csv("train.csv", encoding="utf-8")
test = pd.read_csv("test.csv", encoding="utf-8")
# train
# test

"""**利用【pandas.concat】函數合併資料：方便資料整理，之後再拆分**"""

d1 = pd.DataFrame([[1,2],[3,4]])
d2 = pd.DataFrame([[11,12],[13,14]])
pd.concat([d1,d2],axis=0) #資料往下合併
pd.concat([d1,d2],axis=1) #資料往右合併
# d1[[True,False]]
# d1[[True,True]]

# concat
datas = pd.concat([train, test], axis=0, ignore_index=True)  # 將train / test 資料合併 
datas = datas.drop(["PassengerId", "Survived"], axis=1) #把欄位捨棄
datas

"""=============================================== 以下為資料整理 ==============================================="""

# 篩選操作 Series/DataFrame[跟你的資料筆數依樣多的True/False]
s = datas.isna().sum() # 檢測isna()缺失值 true(1) / false(0)，sum()加總計數，可以知道該資料表有多少缺失值
s[s > 0].sort_values(ascending=False) #篩選，pandas中的sort_values()函數原理類似於SQL中的order by。ascending參數預設為True(升序排序)，如為False表降序

datas["Embarked"].value_counts() # 計算該欄位各數值出現的次數

most = datas["Embarked"].value_counts().idxmax()   # 該欄位最常出現的數值
datas["Embarked"] = datas["Embarked"].fillna(most)  # 缺失值補進最常出現的數值
# 再看一下缺多少
s = datas.isna().sum()
s[s > 0].sort_values(ascending=False)

"""**篩出姓名中的稱謂(Mr./ Mrs./ Miss)：姓名在預測中無意義，但稱謂有，僅篩出稱謂，簡化內容**"""

datas["Name"]

# 用正規表的方法寫
import re
def name(s):
    # .+任意一次以上 , \s*空白零次以上 (.+) \. .+
    # 比.好的 [a-zA-Z\s()']
    pattern = r".+,\s*(.+)\..+"
    # s = "Braund, Mr. Owen Harris"
    pat = re.compile(pattern)
    return pat.match(s).group(1)
datas["Name"].apply(name)

s = "Braund, Mr. Owen Harris"
# s.split(",")[-1] #用逗號切割後，取逗號後面的值
s.split(",")[-1].split(".") #用.切割
s.split(",")[-1].split(".")[0] #用.切割後，取第一字串
# s.split(",")[-1].split(".")[0].strip() # 針對取出的字串用strip濾掉空白/換行

# 用函式的方法寫
def name(s):
    return s.split(",")[-1].split(".")[0].strip() # 把名子轉為list，先用逗號切割，再用.切割，最後用strip濾掉空白/換行
datas["Name"] = datas["Name"].apply(name) # apply對每個值做相同的運算

"""**船艙**"""

datas["Cabin"]

def cabin(c):
    if pd.isna(c):
        return c #如果有缺值就回傳自己(缺值)
    else:
        return c[0] #不然就回傳第一個字
datas["Cabin"] = datas["Cabin"].apply(cabin)
# datas["Cabin"]

dic = datas["Ticket"].value_counts()
datas["Ticket"] = datas["Ticket"].apply(lambda t:dic[t])

#計算各船艙數量
datas["Cabin"].value_counts()

"""**數列欄位需用median函數取中位數**"""

med = datas.median().drop("Pclass") # Pclass是類別非數列，故須排除
datas = datas.fillna(med) #用fillna函數在缺值欄位中填入中位數
# 再看一下缺多少
s = datas.isna().sum()
s[s > 0].sort_values(ascending=False)

c = datas["Name"].value_counts()
# c 
# [c > 50] #印出C發現共有18種稱謂，前四多稱謂值大於61，第五順位稱謂為8人次，故隨意取50為切點
c = c[c > 50]  
def name2(n):
    if n in c:
        return n #如果為前四大稱謂，回傳稱謂名稱
    else:
        return None #否則填入None
datas["Name"] = datas["Name"].apply(name2)

"""* 類別欄位要用進行one hot encode：將類別欄位每種數值都看成一種型態(如問卷統計)，如該類別欄位可以大小概念解釋，則可選擇不做
* get_dummies函數可以自動將類別欄位依型態拆分並給欄位虛擬名稱
"""

datas = pd.get_dummies(datas)
datas = pd.get_dummies(datas, columns=["Pclass"]) # 因為Pclass欄位被視為數列，所以要手動加回
datas

# 預測後想提高預測值，可另進行合理的欄位增加，如增加Family欄位
datas["Family"] = datas["SibSp"] + datas["Parch"]
datas

"""=============================================== 資料整理後再切分資料===============================================

**.iloc：取整列**
"""

# .iloc -> [1st row, 2nd row, 3rd....]
x = datas.iloc[:len(train)] # 取到len(train)這個數值的列
y = train["Survived"]  # 從 x 推論 y (是否存活)
x_predict = datas.iloc[len(train):] # 從len(train)這個數值取到最後一列
# x_predict

"""**採用隨機森林：聚集決策樹(預設為10組決策樹)，先利用GridSearchCV找出隨機森林的最佳參數值(會自動執行多次)**"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 設定參數範圍
params = {
    "n_estimators":range(21, 200, 2),
    "max_depth":range(5, 11)
}
clf = RandomForestClassifier() 

# GridSearchCV找出最佳參數值
search = GridSearchCV(clf, params, cv=10, n_jobs=-1) 
search.fit(x, y)
print(search.best_score_)
print(search.best_params_)

import numpy as np
from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=145, max_depth=7) # n_estimators：幾個決策樹的平均，max_depth：決策樹幾層，先帶入GridSearchCV找出最佳參數值，再微調(固定一值調另一值)
scores = cross_val_score(clf, x, y, cv=10, n_jobs=-1) # cv(預設為10)：資料要切幾等分，每等分個別輪流當驗證資料，餘當訓練。n_jobs=-1：-1表示用電腦CPU全部的效能
print(scores)
print(np.average(scores))

clf = RandomForestClassifier(n_estimators=51, max_depth=7)
clf.fit(x, y) # 建模
pre = clf.predict(x_predict) # 驗證
result = pd.DataFrame({
    "PassengerId":test["PassengerId"],
    "Survived":pre
})
result.to_csv("rf.csv", encoding="utf-8", index=False)
result

"""將K值標準化(最小最大值標準化)，使其落於0~1之間
* 將資料縮放至[0, 1]間。訓練過程: fit_transform() ，資料型態是array
"""

# !!!! 11/15
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# datas_scale = pd.DataFrame(scaler.fit_transform(datas)) #透過fit_transform轉換數值為0~1，再用pd.DataFrame轉為表格

datas_scale = pd.DataFrame(scaler.fit_transform(datas),
                           columns=datas.columns) #加上表頭
datas_scale

# .iloc -> [1st row, 2nd row, 3rd....]
x_scale = datas_scale.iloc[:len(train)]
y = train["Survived"]
x_predict_scale = datas_scale.iloc[len(train):]
# x_predict_scale

"""演算法-K Neighbors Classifier
* 透過鄰居(離自己最近的點)來推斷所屬的類別
* 比較樣本之間的特徵遠近；相似的樣本，特徵之間的值應該都是相近的
* 定義：如果一個樣本在特徵空間中 與K個最相似(即特徵空間中最鄰近)的樣本中的大多數屬於同一個類別，則該樣本也屬於這個類別
* 使用KNN算法時需將數據集做標準化使其更為穩定(使用sklearn.neighbors.KNeighborsClassifier())
"""

from sklearn.neighbors import KNeighborsClassifier
params = {
    "n_neighbors":range(5, 100)  #n_neighbors：選擇查詢的鄰居數，預設值為5
}
clf = KNeighborsClassifier()
search = GridSearchCV(clf, params, cv=10, n_jobs=-1)
search.fit(x_scale, y)
print(search.best_score_)
print(search.best_params_)

clf = KNeighborsClassifier(n_neighbors=11)
clf.fit(x_scale, y)
pre = clf.predict(x_predict_scale)
result = pd.DataFrame({
    "PassengerId":test["PassengerId"],
    "Survived":pre
})
result.to_csv("knn.csv", encoding="utf-8", index=False)
result