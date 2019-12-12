# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:31:18 2019

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import Perceptron

df = pd.read_csv("iris.csv", header=None)
print(df.tail())
#1-100行目の温煦的変数の抽出
y = df.iloc[0:100, 4].values
#Iris-setosaを-1，Iris-virginicaを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)
#1-100行目の1,3列目の抽出
X = df.iloc[0:100, [0, 2]].values
#品種setosaをプロット（赤の〇）
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
#品種versicolorのプロット
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
#軸らラベルの設定
plt.xlabel('sepal length [cm]')
plt.ylabel('setal length [cm]')
#凡例の設定（左上に配置）
plt.legend(loc='upper left')
#表示
plt.show()

#パーセプトロンオブジェクト作成
ppn = Perceptron.Perceptron(eta=0.1, n_iter=10)
#トレーニングデータへのモデルの適合
ppn.fit(X, y)
#エポックと誤分類の関係の折れ線グラフをプロット
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
#軸ラベルの設定
plt.xlabel('Epochs')
plt.ylabel('Number of update')
#図の表示

plt.show()
