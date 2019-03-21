from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
import pandas as pd

# 初始化 reader
reader = Reader(rating_scale=(1, 5), line_format='user item rating timestamp')

# 初始化 Dataset
df_data = pd.read_csv('./ratings.csv', usecols=['userId', 'movieId', 'rating'])
data = Dataset.load_from_df(df_data, reader)


train_data, test_data = train_test_split(data, test_size=0.2)  # 实际调用 .split.ShuffleSplit()


model = SVD(n_factors=30)
model.fit(train_data)

print(model.pu.shape)
print(model.qi.shape)
model.predict(3,5)
model.get_neighbors(3, 2)