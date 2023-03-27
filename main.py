# -*- CODING: UTF-8 -*-
# @time 2023/3/24 23:36
# @Author tyqqj
# @File main.py
# @Aim: SVD压缩图片

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


def load_one_data(path):
    if not os.path.exists(path):
        print("文件不存在")
        return None
    img = cv2.imread(path)
    img = img.astype(np.int32)
    return img


def svd_compress(data, k):
    u, s, v = np.linalg.svd(data)
    u_k = u[:, :k]
    s_k = np.diag(s[:k])
    v_k = v[:k, :]
    data_k = np.dot(np.dot(u_k, s_k), v_k)
    return data_k


def SVD_3C(data, k):
    data_r = data[:, :, 0]
    data_g = data[:, :, 1]
    data_b = data[:, :, 2]
    data_r_k = svd_compress(data_r, k)
    data_g_k = svd_compress(data_g, k)
    data_b_k = svd_compress(data_b, k)
    data_k = np.dstack((data_r_k, data_g_k, data_b_k))
    return data_k


def save_image(image_data, path):
    cv2.imwrite(path, image_data)


if __name__ == '__main__':
    input_path = input("请输入图片路径：")
    if not os.path.exists(input_path):
        input_path = 'D:\Data\simple_img\img_1.png'
    k = int(input("请输入压缩维度："))
    if k < 1:
        k = 1
    output_path = input("请输入保存路径（包含文件名和扩展名）：")
    if not os.path.exists(output_path):
        # 取出input_path的path和文件名
        path = os.path.split(input_path)[0]
        file_name = os.path.split(input_path)[1]
        # 取出文件名和扩展名
        file_name = os.path.splitext(file_name)[0]
        file_ext = os.path.splitext(input_path)[1]
        output_path = path + '\\' + file_name + '_k_' + str(k) + file_ext

    data = load_one_data(input_path)
    if data is not None:
        data_k = SVD_3C(data, k)
        plt.imshow(data_k / 255)
        plt.show()
        save_image(data_k, output_path)
        print(f"压缩后的图片已保存在：{output_path}")
    else:
        print("图片加载失败，请检查路径")
