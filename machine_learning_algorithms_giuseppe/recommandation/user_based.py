import numpy as np

nb_users = 1000
users = np.zeros(shape=(nb_users, 4))

for i in range(nb_users):
    """
    假设有 1000 个用户，每个用户有 4 个特征，每个特征随机赋值
    """
    users[i, 0] = np.random.randint(0, 4)
    users[i, 1] = np.random.randint(0, 2)
    users[i, 2] = np.random.randint(0, 5)
    users[i, 3] = np.random.randint(0, 5)

nb_products = 20
user_products = np.random.randint(0, nb_products, size=(nb_users, 5))

from sklearn.neighbors import NearestNeighbors

# 选择欧式半径等于 2 时的 20 个邻居
nn = NearestNeighbors(n_neighbors=20, radius=2.0)
nn.fit(users)

test_user = np.array([2, 0, 3, 2])
d, neighbors = nn.kneighbors(test_user.reshape(1, -1))
print(neighbors)

suggested_products = []
for n in neighbors:
    for products in user_products[n]:
        for product in products:
            if product != 0 and product not in suggested_products:
                suggested_products.append(product)

print(suggested_products)
