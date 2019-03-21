def item_cosine_similarity(train):
    """

    :param train: 
    :return:
    """
    import math

    C = dict()
    N = dict()

    for u, items in train.items():
        for i in items.keys():
            if i not in N.keys():
                N[i] = 0
            N[i] += items[i] * items[i]

            for j in items.keys():
                if i == j:
                    continue
                if i not in C.keys():
                    C[i] = dict()
                if j not in C[i].keys():
                    C[i][j] = 0
                # 当用户同时购买了 i 和 j, 则加评分乘积
                C[i][j] += items[i] * items[j]

    W = dict()  # 书本相似度分数
    for i, related_items in C.items():
        if i not in W.keys():
            W[i] = dict()
        for j, cij in related_items.items():
            W[i][j] = cij / (math.sqrt(N[i] * math.sqrt(N[j])))
    return W


if __name__ == '__main__':
    item_cosine_similarity(None)
