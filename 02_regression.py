
import pandas as pd # 潛規則要改名為pd
from sklearn.datasets import load_boston 
boston = load_boston() # 準備資料，目的：預測該區域的平均房價
df = pd.DataFrame(boston["data"],
            columns=boston["feature_names"])
df["ans"] = boston["target"]
df.to_csv("boston.csv", encoding="utf-8", index=False)
df #比較漂亮格式印出來 jupinotebook常用，要寫在最後一行

"""開始丟到sklearn前,把所有東西轉換成numpy array"""

# 開始丟到sklearn前, 把所有東西轉換成numpy array(好習慣)
import numpy as np
from sklearn.model_selection import train_test_split
# x: 輸入 y: 輸出
y = df["ans"]
y = np.array(y)
x = df.drop(["ans"], axis=1)
x = np.array(x)
# 90% x, 10% x, 90% y, 10% y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
# train_test_split([1, 2, 3, 4], 
#                  ["a", "b", "c", "d"],
#                  [101, 102, 103, 104],
#                  test_size=0.25)

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(max_depth=5) #篩分到第幾層
reg.fit(x_train, y_train)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(15, 15))
plot_tree(reg, 
     feature_names=boston["feature_names"],
     max_depth=2, #秀圖只展現幾層
     filled=True)

# regression不會有accuracy_score

reg.predict(x_test) # 預測模型

from sklearn.metrics import r2_score
pre = reg.predict(x_test)
r2_score(y_test, pre) # 驗證模型準確性