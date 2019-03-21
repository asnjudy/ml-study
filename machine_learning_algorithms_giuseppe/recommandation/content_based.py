import numpy as np
from sklearn.neighbors import NearestNeighbors

nb_items = 1000
items = np.zeros(shape=(nb_items, 4))

for i in range(nb_items):
    """
    假设有 1000 个用户，每个用户有 4 个特征，每个特征随机赋值
    """
    items[i, 0] = np.random.randint(0, 100)
    items[i, 1] = np.random.randint(0, 100)
    items[i, 2] = np.random.randint(0, 100)
    items[i, 3] = np.random.randint(0, 100)

# 在半径5.0内，找出与指定样本点最近的10个邻居
# 未指定距离度量指标，默认使用欧氏距离
# euclidean, hamming, jaccard
nn = NearestNeighbors(n_neighbors=10, radius=5.0, metric='jaccard')
nn.fit(items)

test_product = np.array([15, 60, 28, 73])
distances, suggestions = nn.radius_neighbors(test_product.reshape(1, -1), radius=20)


def distance_validate():
    from scipy.spatial.distance import jaccard
    target_items = items[suggestions[0], :]
    target_distances = []
    for target_item in target_items:
        d = jaccard(test_product, target_item)
        target_distances.append(d)
    print(distances[0] == target_distances)


distance_validate()
print(distances[0].shape)
