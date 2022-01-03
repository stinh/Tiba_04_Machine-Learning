
# 準備資料
import pandas as pd  # 做表比較好看
from sklearn.datasets import load_iris
iris = load_iris()  # 載入資料 ，想看原始資料可以 print(iris)
df = pd.DataFrame(iris["data"],
    columns=iris["feature_names"])  # 資料整理成表格
df # 印出df

"""**已知分幾群**"""

from sklearn.cluster import KMeans
cluster = KMeans(n_clusters=3) # 設定分群數，要分幾群：分3群，0、1、2
cluster.fit(iris["data"])   # fit:進行分群

cluster.labels_ # 進行分群標示，每一個點屬於哪一群

df["label"] = cluster.labels_  # 進行分群標示，每一個點屬於哪一群，並以表格呈現
df

cluster.cluster_centers_ # 產生每一群虛擬中心點：分群後該群真正的中心位置(不是樣本點上)

"""**不知道分幾群**"""

from sklearn.metrics import silhouette_score
for testk in range(2,20):  # 不知道要分幾群，所以試試看 分別分成2~19群看看
  testc = KMeans(n_clusters=testk) # 設定分群數
  testc.fit(iris["data"]) #進行分群
  s = silhouette_score(iris["data"],testc.labels_) #做分群標示後進行silhouette_score計算
  
  print(testk,":",s) # 印出分幾群，係數多少，係數0.5以上都是可選擇的選項係數0.5以上都是可選擇的選項(統計是評估標準，篩出可選擇選項)

"""**直接將資料轉為散佈圖來觀察**"""

iris  # 看iris資料的詳細內容

'''
從詳細資料可以知道feature_names': ['sepal length (cm)',  'sepal width (cm)',  'petal length (cm)',  'petal width (cm)']
target_names': array(['setosa', 'versicolor', 'virginica'], dtype='<U10')} ，表示【0】：setosa、【1】：versicolor、【2】：virginica
'''

import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(x = df["sepal length (cm)"], # 做散佈圖X軸
        y = df["petal length (cm)"], # 做散佈圖Y軸
        hue = iris["target"])  # 對iris做散佈圖，並加上標示(顏色區分)

import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(x = df["sepal width (cm)"], # 做散佈圖X軸
        y = df["petal width (cm)"], # 做散佈圖Y軸
        hue = iris["target"])  # 對iris做散佈圖，並加上標示(顏色區分)