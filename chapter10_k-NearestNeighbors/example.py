###k近邻算法

#在散点图中为某个数据分类, 以它围一个圈, 分到圈内数量最多的一类
##kNN算法用于分类  回归

#使用kNN算法的步骤

#1、数据准备: 
# 首先要有数据集, 例如二手房屋的售价和相关特征的数据集, 我们还要将数据做标准化(将数据按比例缩放, 以便每个特征的均值为0, 标准差为1)和归一化(将特征值缩放到一个特定区域), 以消除某些数据尺度上的差异, 使得每个数据在距离计算中有着相同的权重

#2、距离度量: 
# 计算目标点与邻居的距离((使用较多)欧几里得距离(空间距离); 
# 曼哈顿距离(垂直路径长度, 不包含斜线)适用于高维数据分析\网格状或离散环境的场景中; 
# 余弦相似度(A与B的点积比上A和B的范数(向量的长度)的乘积)), 对向量长度不敏感只关心方向, 在文本分析, 用户偏好分析和其他高维空间的数据中使用较为广泛(例如根据文本词频向量计算余弦相似度, 此项技术用于搜索引擎, 推荐系统, 文本挖掘等场景)

#3、确定k值: 
# k值直接影响拟合的效果, k过小, 容易受到噪声数据的干扰导致过拟合, k过大, 容易忽略局部特征, 导致欠拟合; 
# 合适的k值一般通过对不同k值的交叉验证选出

#4、模型训练: kNN算法本质是一种基于示例的算法, 或者说是懒惰学习, 没有明确的模型训练过程, 训练的数据已经存储好了, 算法会在训练数据集中, 找到与目标点最近的K个邻居, 不过这样的代价就是它在预测阶段需要更多的计算资源和时间, 如果训练数据集非常大, 存储和搜索邻居的成本也会非常高

###使用KNN的具体案例
"""
以下为五个用户对五部电影的观影感受评分, 0表示未观影:
    Psycho  Rocky   Platoon Goldfinger  Jaws
A:  5       3       0       0           2
B:  0       4       4       3           1
C:  2       0       5       0           0
D:  0       0       4       5           0
E:  3       2       0       2           3
现在有另一个人, 需要推荐电影(从他没看过的两部电影中推荐一部): 
F:  3       0       5       0           3
"""
##实现模拟功能
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# 模拟的数据: 5部电影, 五个用户
ratings = np.array([
    [5, 3, 0, 0, 2],
    [0, 4, 4, 3, 1],
    [2, 0, 5, 0, 0],
    [0, 0, 4, 5, 0],
    [3, 2, 0, 2, 3]
])

# 使用KNN找到与当前用户最相似的用户
# 这里我们设定为5个最近邻居来得到所有5位用户的距离
knn = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute') # n_neighbors为k值, metrics为距离度量, algorithm为搜索算法
knn.fit(ratings)

# 假设当前用户评分是[3, 0, 5, 0, 3]
current_user_rating = np.array([[3, 0, 5, 0, 3]])
distances, indices = knn.kneighbors(current_user_rating)

# 打印距离
for i, (index, distance) in enumerate(zip(indices[0], distances[0])):
    print(f"当前用户与用户{index}的距离为: {distance:.4f}")

# 使用PCA将数据降为2维, 以便进行可视化
pca = PCA(n_components=2)
transformed_ratings = pca.fit_transform(ratings)
transformed_current_user = pca.transform(current_user_rating)

# 使用matplotlib绘制散点图
plt.figure(figsize=(10, 6))

# 绘制所有用户
plt.scatter(transformed_ratings[:, 0], transformed_ratings[:, 1], label='Other Users', s=100)

# 绘制与当前用户最相似的用户
for index in indices[0]:
    plt.scatter(transformed_ratings[index, 0], transformed_ratings[index, 1], label=f'User {index}', s=100)

# 绘制当前用户
plt.scatter(transformed_current_user[:, 0], transformed_current_user[:, 1], marker='*', color='red', s=200, label='Current User')

plt.title('Visualization of User Similarities')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

##实现回归功能
import numpy as np
from sklearn.neighbors import NearestNeighbors

# 模拟的数据: 5部电影, 五个用户
ratings = np.array([
    [5, 3, 0, 0, 2],
    [0, 4, 4, 3, 1],
    [2, 0, 5, 0, 0],
    [0, 0, 4, 5, 0],
    [3, 2, 0, 2, 3]
])

# 使用KNN找到与当前用户最相似的用户
# 这里选择三个最近的邻居
knn = NearestNeighbors(n_neighbors=3, metric='cosine', algorithm='brute') # n_neighbors为k值, metrics为距离度量, algorithm为搜索算法
knn.fit(ratings)

# 假设当前用户评分是[3, 0, 5, 0, 3]
current_user_rating = np.array([[3, 0, 5, 0, 3]])
distances, indices = knn.kneighbors(current_user_rating)

# 计算平均评分, 但要忽略评分为0的部分
neighbors_ratings = ratings[indices][0] # 提取邻居的评分
sum_ratings = np.sum(neighbors_ratings, axis=0)
count_nonzero = np.count_nonzero(neighbors_ratings, axis=0)

# 避免除以0
mean_ratings = np.divide(sum_ratings, count_nonzero, out=np.zeros_like(sum_ratings, dtype=float), where=count_nonzero != 0)

# 打印出2号和4号电影(索引1和3)的预测评分
print(f"预测的2号电影评分为: {mean_ratings[1]}")
print(f"预测的4号电影评分为: {mean_ratings[3]}")


###KNN算法是机器学习中相对简单和直观的一种算法, 它适用于分类和回归问题; 
# 在一些简单和小型的问题上KNN可能会达到非常好的结果, 但在更复杂和大规模的问题上则需要使用更复杂的算法;
# 对于机器学习学习的内容, KNN算法是一个很好的入门点, 它可以帮助我们理解机器学习中的一些最基础的概念和术语, 比如训练集、测试集、特征、标签、分类、回归等; 
# 继续学习机器学习的内容需要更多的算法和基础的数学的统计知识, 包括但不限于决策树、支持向量机、神经网络以及如何在不同的场景下使用它们, 还要学习如何处理更复杂的问题, 如自然语言处理、图像识别、强化学习等等