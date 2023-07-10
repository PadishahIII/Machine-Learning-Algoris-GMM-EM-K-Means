import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import math

MAXINT = 1e300


#高斯混合模型
class GMM_EM:
    def __init__(self, UCI, iniMode):
        self.iniMode = iniMode
        self.data = []
        if UCI:
            csv = pd.read_csv(
                "./UCIdata.csv",
                header=0,
                #  nrows=2000,
                encoding="gbk")
            self.data = np.array(csv[["a", "c"]])[4000:4400, :]
        else:
            self.data = pd.read_csv("./data.csv",
                                    header=None).values  #由四个高斯分布生成的样本点
        self.K = 4
        self.dim = len(self.data[0, :])
        self.center = np.tile(np.array([[.0, .0]]), (self.K, 1))  #中心点
        self.n = len(self.data)  #样本数量
        self.flags = list(0 for i in range(self.n))  #样本点属于的簇
        self.covs = [
            np.array([[2., 1.], [1., 2.]]),
            np.array([[3., 1.], [1., 4.]]),
            np.array([[2., 1.], [1., 2.]]),
            np.array([[3., 1.], [1., 4.]])
        ]  #四个簇的协方差矩阵
        self.ave = np.array([[1., 2.], [6., 7.], [1., 2.], [1., 2.]])  #均值
        self.alpha = [0.25, 0.25, 0.25, 0.25]  #先验概率
        self.r = np.zeros((self.n, self.K))  #响应度，即后验概率

    #欧氏距离
    @staticmethod
    def cal_distance(list1, list2):
        x = np.mat(list1) - np.mat(list2)
        return np.dot(x, x.T)[0][0]

    #损失：每个点到各自中心点的欧氏距离之和
    def cal_loss(self):
        sum = 0
        for i in range(self.n):
            flag = self.flags[i]
            dis = self.cal_distance(self.data[i, :], self.center[flag, :])
            sum += dis
        return sum

    #k-means
    def k_mean(self):
        if self.iniMode:
            self.init_center()
        else:
            self.init_center_()
        loss0 = self.cal_loss()
        loss1 = loss0
        while True:
            for j in range(self.n):
                min_distance = MAXINT
                min_K = 0
                for k in range(self.K):
                    dis = self.cal_distance(self.data[j, :], self.center[k, :])
                    if dis < min_distance:
                        min_distance = dis
                        min_K = k
                self.flags[j] = min_K
            self.update_center()
            loss0 = loss1
            loss1 = self.cal_loss()
            if np.fabs(loss0 - loss1) < 1e-10:
                break

    #随机选择中心点
    def init_center(self):
        centerIndex = set()
        #生成K个随机数
        for i in range(self.K):
            centerIndex.add(int(math.fabs(random.randint(0, self.n - 1))))
        assert len(centerIndex) == self.K

        i = 0
        for num in centerIndex:
            self.center[i, 0] = self.data[num][0]
            self.center[i, 1] = self.data[num][1]
            i += 1

    #手动初始化
    def init_center_(self):
        self.center[0, 0] = self.data[0][0]
        self.center[0, 1] = self.data[0][1]

        self.center[1, 0] = self.data[50][0]
        self.center[1, 1] = self.data[50][1]

        self.center[2, 0] = self.data[100][0]
        self.center[2, 1] = self.data[100][1]

        self.center[3, 0] = self.data[150][0]
        self.center[3, 1] = self.data[150][1]

    #求各簇的中心点
    def update_center(self):
        for k in range(self.K):
            count = 0
            sum = np.zeros((1, self.dim))
            for i in range(self.n):
                if self.flags[i] == k:
                    count += 1
                    sum += self.data[i, :]
            self.center[k] = sum / count

    #k-means的结果作为初值
    def cal_init(self):
        for i in range(self.K):
            for j in range(self.dim):
                self.ave[i, j] = self.center[i, j]
        cov = []  #每个簇的协方差矩阵
        for k in range(self.K):
            array = []  #所有属于k簇的点
            for i in range(self.n):
                if self.flags[i] == k:
                    array.append(self.data[i, :])
            array = np.array(array)
            aver_k = []  #均值
            for i in range(len(array)):
                aver_k.append(self.ave[k, :])
            aver_k = np.array(aver_k)
            cov.append(np.dot((array - aver_k).T, (array - aver_k)))
        self.covs = np.array(cov)

    #第k个混合成分的概率密度
    def cal_prob(self, j, k):
        #分母
        x = (2 * np.pi)**(self.dim * 1.0 / 2) * np.linalg.det(
            self.covs[k])**0.5
        y = np.exp(-0.5 * np.dot(
            np.dot(
                (self.data[j, :] - self.ave[k]), np.linalg.inv(self.covs[k])),
            (self.data[j, :] - self.ave[k]).T))
        return y / x

    #样本j由第i个高斯混合成份生成的概率
    def cal_mix_gauss_prob(self, i, j):
        #分子
        x = self.alpha[i] * self.cal_prob(j, i)
        #分母
        y = 0
        for k in range(self.K):
            y += self.alpha[k] * self.cal_prob(j, k)

        return x / y

    #第k簇的样本响应度之和
    def cal_res_sum(self, k):
        sum = 0
        for i in range(self.n):
            sum += self.r[i][k]
        return sum

    #高斯混合聚类
    def Mix_Guass(self, round):
        while round > 0:
            round -= 1
            #计算后验概率  E步
            for i in range(self.n):
                for k in range(self.K):
                    self.r[i][k] = self.cal_mix_gauss_prob(k, i)
            for i in range(self.K):
                x = np.zeros((1, self.dim))
                for j in range(self.n):
                    x += self.r[j][i] * self.data[j, :]
                x /= self.cal_res_sum(i)
                #更新均值  M步
                self.ave[i] = x

                y = self.data - np.tile(self.ave[i], (self.n, 1))
                z = np.eye(self.n)
                for j in range(self.n):
                    z[j][j] = self.r[j][i]
                #更新协方差矩阵
                self.covs[i] = np.dot(np.dot(y.T, z), y) / self.cal_res_sum(i)

                self.alpha[i] = self.cal_res_sum(i) / self.n

    #更新簇标记
    def update_flags(self):
        for j in range(self.n):
            k = 0
            max = 0
            for i in range(self.K):
                lamda_j = self.r[j][i]
                if lamda_j > max:
                    max = lamda_j
                    k = i
            self.flags[j] = k

    def updata_center_Gauss(self):
        for k in range(self.K):
            self.center[k] = self.ave[k]

    #可视化
    def draw(self):
        x0 = []
        y0 = []
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        x3 = []
        y3 = []
        for i in range(self.n):
            flag = self.flags[i]
            if flag == 0:
                x0.append(self.data[i][0])
                y0.append(self.data[i][1])
            elif flag == 1:
                x1.append(self.data[i][0])
                y1.append(self.data[i][1])
            elif flag == 2:
                x2.append(self.data[i][0])
                y2.append(self.data[i][1])
            elif flag == 3:
                x3.append(self.data[i][0])
                y3.append(self.data[i][1])
        #plt.legend(loc=2)
        plt.scatter(x0, y0, marker='*', c='orange', label='class1')
        plt.scatter(x1, y1, marker='D', c='lightskyblue', label='class2')
        plt.scatter(x2, y2, marker='x', c='lightgreen', label='class3')
        plt.scatter(x3, y3, marker='o', c='tomato', label='class4')
        plt.scatter(self.center[0, 0],
                    self.center[0, 1],
                    marker='+',
                    color='blue',
                    s=100,
                    label="center")
        for k in range(1, self.K):
            plt.scatter(self.center[k, 0],
                        self.center[k, 1],
                        marker='+',
                        color='blue',
                        s=100)
        plt.xlabel('length')
        plt.ylabel('width')

        plt.legend(loc=2)
        plt.show()


#生成四种高斯分布的数据
def generate_data(file):
    f = open(file, 'w')
    mean0 = [1, 5]
    mean1 = [1, 1]
    mean2 = [5, 1]
    mean3 = [5, 5]
    cov = np.mat([[1, 0], [0, 1]])
    data0 = np.random.multivariate_normal(mean0, cov, 50).T
    data1 = np.random.multivariate_normal(mean1, cov, 50).T
    data2 = np.random.multivariate_normal(mean2, cov, 50).T
    data3 = np.random.multivariate_normal(mean3, cov, 50).T

    plt.scatter(data0[0], data0[1], marker='*', c='orange', label='class1')
    plt.scatter(data1[0],
                data1[1],
                marker='D',
                c='lightskyblue',
                label='class2')
    plt.scatter(data2[0], data2[1], marker='x', c='lightgreen', label='class3')
    plt.scatter(data3[0], data3[1], marker='o', c='tomato', label='class4')
    plt.legend(loc=2)
    plt.title("raw data")
    plt.show()

    for i in range(len(data0.T)):
        line = []
        line.append(data0[0][i])
        line.append(data0[1][i])
        line = ",".join(str(i) for i in line)
        line = line + "\n"
        f.write(line)
    for i in range(len(data1.T)):
        line = []
        line.append(data1[0][i])
        line.append(data1[1][i])
        line = ",".join(str(i) for i in line)
        line = line + "\n"
        f.write(line)
    for i in range(len(data2.T)):
        line = []
        line.append(data2[0][i])
        line.append(data2[1][i])
        line = ",".join(str(i) for i in line)
        line = line + "\n"
        f.write(line)
    for i in range(len(data3.T)):
        line = []
        line.append(data3[0][i])
        line.append(data3[1][i])
        line = ",".join(str(i) for i in line)
        line = line + "\n"
        f.write(line)
    f.close()


UCI = False
iniMode = False  #True:随机选取center

#generate_data("./data.csv")
gmm = GMM_EM(UCI, iniMode)
gmm.k_mean()  #k-means算法
plt.title("K-means")
gmm.draw()
gmm.cal_init()  #用k-means算法的结果作为初值
print("K-means:\n")
print("均值:\n")
print(gmm.ave)
print("协方差：")
print(gmm.covs)
gmm.Mix_Guass(100)  #高斯混合聚类
gmm.update_flags()  #重新划分样本点
gmm.updata_center_Gauss()
plt.title("GMM_EM")
gmm.draw()
print("\n\nMix-Gauss:\n")
print("均值:\n")
print(gmm.ave)
print("协方差：")
print(gmm.covs)
