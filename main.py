# -*- CODING: UTF-8 -*-
# @time 2023/3/24 23:36
# @Author tyqqj
# @File main.py
# @Aim: SVD压缩图片

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

path = './val_data'


def load_one_data(path):
    if not os.path.exists(path):
        print("文件不存在")
        return None
    img = cv2.imread(path)
    img = img.astype(np.int32)
    return img


def svd_compress(data, k):
    u, s, v = np.linalg.svd(data)
    # print("k = " + str(k) + ":", u.shape, s.shape, v.shape)
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
    data_k = np.clip(data_k, 0, 255)  # 像素值限制在0-255之间
    return data_k.astype(np.uint8)


def save_image(image_data, path):
    cv2.imwrite(path, image_data)


def show_imgs(imgs, titles, nl=1, sub_title=None):
    n = len(imgs)
    plt.figure()
    for i in range(n):
        plt.subplot(nl, int((n + 1) / nl), i + 1)
        plt.imshow(imgs[i])
        plt.axis('off')  # 不显示坐标轴
        plt.title(titles[i])
    if sub_title is not None:
        plt.suptitle(sub_title)
    plt.show()


# 展示程序
def show_image(path):
    # 读取图片
    name_lst = os.listdir(path)
    imgs = []
    for name in name_lst:
        imgs.append(load_one_data(path + '/' + name))

    # 在一个窗口中显示多张图片
    show_imgs(imgs, name_lst, 2, 'original')

    # SVD压缩
    k_l = [1, 10, 100]
    for k in k_l:
        print("k = " + str(k) + "压缩开始")
        imgs_k = []
        i = 0
        for img in imgs:
            i += 1
            imgs_k.append(SVD_3C(img, k))
            print("k = " + str(k) + ",", i, "压缩完成")
        show_imgs(imgs_k, name_lst, 2, 'k = ' + str(k))
        print("k = " + str(k) + "展示完成")

    print("展示程序结束")


if __name__ == '__main__':
    show_image(path)
    # input("按任意键退出")
