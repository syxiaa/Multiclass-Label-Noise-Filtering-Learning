import  pandas as pd
from sklearn.datasets import  make_blobs,make_circles,make_moons
from sklearn.cluster import  KMeans
import  numpy as np
from collections import Counter



def create_dataset(dataset_name, noise_rate, n_samples=1500, n_clusters=3):


    if dataset_name == 'moons':
        X1, Y1 = make_moons(n_samples=300, random_state=0, noise=noise_rate)
        X2, Y2 = make_moons(n_samples=525, random_state=0, noise=noise_rate)
        X3, Y3 = make_moons(n_samples=675, random_state=0, noise=noise_rate)
        X = np.vstack((X1, X2, X3))
        y1 = np.array([0] * 300)
        y2 = np.array([1] * 525)
        y3 = np.array([2] * 675)  # 生成标签数组
        Y = np.hstack((y1, y2, y3))
        # plt.scatter(X[:, 0], X[:, 1], c=Y)
        # plt.show()

        # 用kmeans聚成三类，将原始数据集划分成三类。
        y_pred = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(X)
        # print(Counter(y_pred))
        # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        # plt.show()

        x = pd.DataFrame(X)
        y_pred = pd.DataFrame(y_pred)
        moons = pd.concat([y_pred, x], axis=1)  # 按列拼接datafram/series
        l0 = []
        l1 = []
        l2 = []
        moons.columns = ['0', '1', '2']  # 修改df的列名字
        for index, row in moons.iterrows():  # 采样
            if row[0] == 0:
                if len(l0) < 200:
                    l0.append(row)
                    continue
            elif row[0] == 1:
                if len(l1) < 350:
                    l1.append(row)
                    continue

            elif row[0] == 2:
                if len(l2) < 450:
                    l2.append(row)
                    continue

        print(len(l0), '\t', len(l1), '\t', len(l2))
        data = l0 + l1 + l2
        data = pd.DataFrame(data)  # 列表转化dataframe
        data = data.as_matrix()  # dataframe转为ndarra
        # plt.scatter(data[:, 1], data[:, 2], c=data[:, 0])
        # plt.show()

        moons = data  # dataframe转为ndarrar
        X = moons[:, 1:]
        Y = moons[:, 0]
        return X, Y


    elif dataset_name == 'circles':
        X1, Y1 = make_circles(n_samples=300, noise=noise_rate, random_state=0)
        X2, Y2 = make_circles(n_samples=525, noise=noise_rate, random_state=0)
        X3, Y3 = make_circles(n_samples=675, noise=noise_rate, random_state=0)
        X = np.vstack((X1, X2, X3))
        y1 = np.array([0] * 300)
        y2 = np.array([1] * 525)
        y3 = np.array([2] * 675)  # 生成标签数组
        Y = np.hstack((y1, y2, y3))
        # plt.scatter(X[:, 0], X[:, 1], c=Y)
        # plt.show()

        # 用kmeans聚成三类，将原始数据集划分成三类
        y_pred = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(X)
        print(Counter(y_pred))
        # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        # plt.show()

        x = pd.DataFrame(X)
        y_pred = pd.DataFrame(y_pred)
        circles = pd.concat([y_pred, x], axis=1)  # 按列拼接datafram/series
        l0 = []
        l1 = []
        l2 = []
        circles.columns = ['0', '1', '2']  # 修改df的列名字
        for index, row in circles.iterrows():  # 采样
            if row[0] == 0:
                if len(l0) < 200:
                    l0.append(row)
                    continue
            elif row[0] == 1:
                if len(l1) < 350:
                    l1.append(row)
                    continue

            elif row[0] == 2:
                if len(l2) < 450:
                    l2.append(row)
                    continue

        print(len(l0), '\t', len(l1), '\t', len(l2))
        data = l0 + l1 + l2
        data = pd.DataFrame(data)  # 列表转化dataframe
        data = data.as_matrix()  # dataframe转为ndarra
        # plt.scatter(data[:, 1], data[:, 2], c=data[:,0])
        # plt.show()

        circles = data
        X = circles[:, 1:]
        Y = circles[:, 0]
        return X, Y



    elif dataset_name == 'blobs':
        X1, y11 = make_blobs(n_samples=200, centers=1,
                             random_state=18, cluster_std=0.8, center_box=(-3, 3))
        X2, y22 = make_blobs(n_samples=350, centers=1,
                             random_state=1, cluster_std=0.9, center_box=(-1, 3))
        X3, y33 = make_blobs(n_samples=450, centers=1,
                             random_state=3, cluster_std=1, center_box=(2, 4))
        X = np.vstack((X1, X2, X3))
        y1 = np.array([0] * 200)
        y2 = np.array([1] * 350)
        y3 = np.array([2] * 450)
        y = np.hstack((y1, y2, y3))
        # plt.scatter(X[:,0],X[:,1],c=y)
        # plt.show()

        return X, y  # 样本和标签是分开的