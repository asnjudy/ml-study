from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate

# 定义数据格式
reader = Reader(line_format='user item rating timestamp', sep='::')

# 使用reader格式从u.data文件中读取数据
data = Dataset.load_from_file('./ml-1m/ratings.dat', reader=reader)

data.split(n_folds=5)

model = SVD()

print(cross_validate(model, data, measures=['RMSE', 'MAE'], cv=3))


model.predict()