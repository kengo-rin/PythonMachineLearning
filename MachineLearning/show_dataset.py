# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:31:18 2019

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("iris.csv", header=None)
print(df.tail())
#1-100行目の温煦的変数の抽出
y = df.iloc[0:100, 4].values
#Iris-setosaを-1，Iris-virginicaを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)
#1-100行目の1,3列目の抽出
X = df.iloc[0:100, [0, 2]].values
#品種setosaをプロット（赤の〇）
plt.scatter(X[:50, 0],)

