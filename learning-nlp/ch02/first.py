import numpy as np

matrix = np.array([
    ['5', '10', 15],
    [20, 25, 30],
    [35, 40, '']
])

vector = np.array(['1', '2', '3', '3.128'])
vector_float = vector.astype(float)

matrix = np.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 10]
])

print(matrix)
print(matrix.sum(axis=1))
print(matrix.sum(axis=0))
