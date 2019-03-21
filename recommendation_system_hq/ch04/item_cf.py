import math


def item_similarity(train):
    """
    ItemCF 协同过滤算法
    :param train: dict
    :return:
    """
    C = dict()
    N = dict()

    for u, items in train.items():
        for i in items.keys():
            if i not in N.keys():
                N[i] = 0
            N[i] += 1

            for j in items.keys():
                if i == j:
                    continue
                if i not in C.keys():
                    C[i] = dict()
                if j not in C[i].keys():
                    C[i][j] = 0
                # 当用户同时购买了 i 和 j, 则加 1
                C[i][j] += 1

    W = dict()
    for i, related_items in C.items():
        if i not in W.keys():
            W[i] = dict()
        for j, cij in related_items.items():
            W[i][j] = cij / math.sqrt(N[i] * N[j])
    return W


if __name__ == '__main__':
    train_data = {
        'A': {'i1': 1, 'i2': 1, 'i4': 1},
        'B': {'i1': 1, 'i4': 1},
        'C': {'i1': 1, 'i2': 1, 'i5': 1},
        'D': {'i2': 1, 'i3': 1},
        'E': {'i3': 1, 'i5': 1},
        'F': {'i2': 1, 'i4': 1}
    }
    W = item_similarity(train_data)
    print(W)
