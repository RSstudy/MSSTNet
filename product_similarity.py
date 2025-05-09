import torch
import numpy as np
import scipy.io as sio
from utility import load_HSI, hyperVca, load_data, reconstruction_SADloss
import os
import random
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
# Calculate the cosine similarity to get similarity matrix
# The closer the cosine distance is to 1, the greater the similarity value. However, in this version of the code,
# the cosine distance is subtracted by 1 and multiplied by 1000, initially to better observe the difference between similarities.
# At this point in the code, the smaller the value in the similarity matrix, the more similar it is.
# You can operate the similarity matrix as follows: similarity_matrix = 1 - similarity_matrix/1000;
# And change the mapping interval to: Mapping_ranges = [(2, 1.8),(1.8, 1.6),(1.6, 1.4),(1.4, 1.2),(1.2, 1),
# (1, 0.8),(0.8, 0.6),(0.6, 0.4),(0.4, 0.2),(0.2, 0.01)].
# This corresponds to the explanation in the paper. It's the same in theory.
# The two are only different in operation, and there is no essential difference.
# 计算余弦相似度以获得相似度矩阵
# 余弦距离越接近1，相似度值越大。但在这个版本的代码中，
# 余弦距离减去1并乘以1000，最初是为了更好地观察相似度之间的差异。
# 在代码的这一点上，相似度矩阵中的值越小，表示越相似。
# 可以按以下方式操作相似度矩阵：similarity_matrix = 1 - similarity_matrix/1000;
# 并更改映射间隔为：Mapping_ranges = [(2, 1.8),(1.8, 1.6),(1.6, 1.4),(1.4, 1.2),(1.2, 1),
# (1, 0.8),(0.8, 0.6),(0.6, 0.4),(0.4, 0.2),(0.2, 0.01)].
# 这与论文中的解释相对应。理论上是一样的。
# 两者只是在操作上不同，在本质上没有区别。

seed = 1
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datasetnames = {'Urban': 'Urban4',
                'Samson': 'Samson',
                'dc': 'DC',
                'Jasper': 'Jasper',
                'sy30': 'sy30',

                }
dataset = "sy30"  # Replace the data set here
hsi = load_HSI("Datasets/" + datasetnames[dataset] + ".mat")
data = hsi.array()
endmember_number = hsi.gt.shape[0]
col = hsi.cols
band_number = data.shape[1]


original_HSI = torch.from_numpy(data)
original_HSI = torch.reshape(original_HSI.T, (band_number, col, col))
image = np.array(original_HSI)

similarity_matrix = np.zeros_like(image)
for i in tqdm(range(image.shape[1])):
    for j in range(image.shape[2]):
        center_pixel = image[:, i, j]
        neighbors = []
        # Get the values of the four surrounding pixels
        if i > 0:
            neighbors.append(image[:, i - 1, j])
        if i < image.shape[1] - 1:
            neighbors.append(image[:, i + 1, j])
        if j > 0:
            neighbors.append(image[:, i, j - 1])
        if j < image.shape[2] - 1:
            neighbors.append(image[:, i, j + 1])
        a = np.array(neighbors)
        if neighbors:
            # 计算中心像素和邻域像素之间的余弦距离
            similarities = cosine_similarity(center_pixel.reshape(1, -1), np.array(neighbors))
            # 计算中心像素和邻域像素之间的欧氏距离
            distances = euclidean_distances(center_pixel.reshape(1, -1), np.array(neighbors))
            distances = np.exp(-distances ** 2)
            # print(distances)
            similarity_matrix[:, i, j] = (1 - np.mean(similarities + distances)) * 1000

if dataset == 'Samson':
    sio.savemat('similarity_samson.mat', {'samson': similarity_matrix})
elif dataset == 'Urban':
    sio.savemat('similarity_urban.mat', {'urban': similarity_matrix})
elif dataset == 'dc':
    sio.savemat('similarity_dc.mat', {'dc': similarity_matrix})
elif dataset == 'Jasper':
    sio.savemat('similarity_Jasper.mat', {'Jasper': similarity_matrix})
elif dataset == 'sy30':
    sio.savemat('similarity_sy30.mat', {'sy30': similarity_matrix})