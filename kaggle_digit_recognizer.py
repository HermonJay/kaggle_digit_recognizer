# Hermon Jay, 15-10-2017
# klasifikasi digit tulisan tangan dengan
# neural network

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# model
tree = DecisionTreeClassifier()

# data
mnist_train = pd.read_csv('train.csv').as_matrix()
mnist_test = pd.read_csv('test.csv').as_matrix()

# masukan X dan keluaran Y
X = mnist_train[:, 1:]
Y = mnist_train[:, 0]
X_test = mnist_test[:,:]

# latih classifier
tree.fit(X,Y)

# prediksi
Y_pred = tree.predict(X_test)

# cetak file submission
ImageId = np.arange(1,28001,1)
Y_pred = np.int64(Y_pred)
submission = pd.DataFrame({
        "ImageId": ImageId,
        "Label": Y_pred
    })
submission.to_csv('submission.csv', index = False)