import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img_eg = mpimg.imread('./picture.jpg')
print(img_eg.shape)

# 奇异值分解
img_temp = img_eg.reshape(582, 1024 * 3)
U, Sigma, VT = np.linalg.svd(img_temp)

# 取前60个奇异值
sval_nums = 60
img_restruct1 = (U[:, 0:sval_nums]).dot(np.diag(Sigma[0:sval_nums])).dot(VT[0:sval_nums, :])
img_restruct1 = img_restruct1.reshape(582, 1024, 3)

# 取前120个奇异值
sval_nums = 120
img_restruct2 = (U[:, 0:sval_nums]).dot(np.diag(Sigma[0:sval_nums])).dot(VT[0:sval_nums, :])
img_restruct2 = img_restruct2.reshape(582, 1024, 3)

fig, ax = plt.subplots(1, 3, figsize=(24, 32))

ax[0].imshow(img_eg)
ax[0].set(title="src")
ax[1].imshow(img_restruct1.astype(np.uint8))
ax[1].set(title="nums of sigma = 60")
ax[2].imshow(img_restruct2.astype(np.uint8))
ax[2].set(title="nums of sigma = 120")
