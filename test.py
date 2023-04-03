# -*- CODING: UTF-8 -*-
# @time 2023/4/3 19:44
# @Author tyqqj
# @File test.py


from sklearn import datasets, svm, metrics
from skimage.feature import hog
import numpy as np

# 加载 MNIST 数据集
digits = datasets.load_digits()

# 分割数据集为训练集和测试集
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
train_data = data[:int(0.8*n_samples)]
test_data = data[int(0.8*n_samples):]
train_labels = digits.target[:int(0.8*n_samples)]
test_labels = digits.target[int(0.8*n_samples):]


# 定义 HOG 特征提取器
hog_feature = lambda img: hog(img, orientations=10, pixels_per_cell=(4, 4),
                              cells_per_block=(1, 1), visualize=False)

# 提取特征
train_features = np.array([hog_feature(img) for img in train_data])
test_features = np.array([hog_feature(img) for img in test_data])


# 创建 SVM 分类器
classifier = svm.SVC()

# 训练分类器
classifier.fit(train_features, train_labels)

# 预测测试集
predicted = classifier.predict(test_features)

# 输出准确率
accuracy = metrics.accuracy_score(test_labels, predicted)
print("准确率：", accuracy)
