# coding:utf8
from scikits.crab.models import MatrixPreferenceDataModel

user_item_matrix = {
    1: {1: 2, 2: 5, 3: 3},
    2: {1: 5, 4: 2},
    3: {2: 3, 4: 5, 3: 2},
    4: {3: 5, 5: 1},
    5: {1: 3, 2: 3, 4: 1, 5: 3}
}


model = MatrixPreferenceDataModel(user_item_matrix)


""""
选择一个度量，构建距离矩阵
"""
from scikits.crab.similarities import UserSimilarity
from scikits.crab.metrics import euclidean_distances
from scikits.crab.recommenders.knn import UserBasedRecommender


similarity_matrix = UserSimilarity(model, euclidean_distances)
recommender = UserBasedRecommender(model, similarity_matrix, with_preference=True)

