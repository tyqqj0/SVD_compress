# -*- CODING: UTF-8 -*-
# @time 2023/3/24 23:36
# @Author tyqqj
# @File main.py
# @Aim: SVD压缩图片

import numpy as np
import sympy as sp
import os
import matplotlib.pyplot as plt

import cv2


def load_data(path):
    """
    加载数据
    :param path: 数据路径
    :return: 数据
    """
    data = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        img = cv2.imread(file_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(-1)
        data.append(img)
    return np.array(data)


def load_one_data(path):
    """
    加载一张图片
    :param path: 图片路径
    :return: 图片
    """
    if not os.path.exists(path):
        print("文件不存在")
        return None
    img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 转换为整形
    img = img.astype(np.int32)
    # img = img.reshape(-1)
    return img


def svd_compress(data, k):
    """
    SVD压缩
    :param data: 数据
    :param k: 压缩后的维度
    :return: 压缩后的数据
    """
    # print("原始数据维度：", data.shape)
    # 压缩
    u, s, v = np.linalg.svd(data)
    print("压缩前的数据维度：", u.shape, s.shape, v.shape)
    # 重构
    u_k = u[:, :k]
    s_k = np.diag(s[:k])
    v_k = v[:k, :]
    print("压缩后的数据维度：", u_k.shape, s_k.shape, v_k.shape)
    data_k = np.dot(np.dot(u_k, s_k), v_k)
    # print("压缩后的数据维度：", data_k.shape)
    return data_k


def SVD_3C(data, k):
    """
    SVD压缩
    :param data: 数据
    :param k: 压缩后的维度
    :return: 压缩后的数据
    """
    print("原始数据维度：", data.shape)
    # 合成三通道
    data_r = data[:, :, 0]
    data_g = data[:, :, 1]
    data_b = data[:, :, 2]
    # 压缩
    data_r_k = svd_compress(data_r, k)
    data_g_k = svd_compress(data_g, k)
    data_b_k = svd_compress(data_b, k)
    # 合成三通道
    data_k = np.dstack((data_r_k, data_g_k, data_b_k))
    return data_k


if __name__ == '__main__':
    """
    主函数
    :return: None
    """
    # 加载数据
    data = load_one_data('D:\Data\simple_img\img_1.png')
    # 压缩数据
    data_k = SVD_3C(data, 1)
    # 重构数据
    # data_k = data_k.reshape(200, 200)
    print("data_k数据类型：", data_k.dtype)
    # 显示图片
    plt.imshow(data_k / 255)
    plt.show()
