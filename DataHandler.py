import pickle
import numpy as np
from scipy.sparse import coo_matrix, issparse # 导入 issparse 函数
from Params import args 
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
from collections import defaultdict
from tqdm import tqdm
import random

'''
1、初始化文件路径
2、加载数据：通过 loadOneFile 方法加载训练集和测试集的稀疏矩阵，并使用 PyTorch 的 Dataset 对象封装数据。
3、特征加载：加载图像、文本特征，并提供相应的 PyTorch 张量表示。
4、数据预处理：对训练数据矩阵进行归一化，并将其转换为适用于 PyTorch 的稀疏张量。
5、负采样：在训练数据集中进行负样本采样，用于训练推荐模型。
6、扩散数据处理：处理扩散过程的数据，以便于后续的图模型计算。

可能的改进/优化：
1、负采样优化：negSampling 方法在循环中每次都会进行随机选择，这可能是性能瓶颈。可以考虑采用更高效的负采样算法。
2、数据预处理优化：如果数据集非常大，可能需要对数据加载进行并行化处理，减少 I/O 阻塞。
3、内存管理：加载大规模数据集时，可以通过批量加载数据和利用稀疏矩阵减少内存开销。

这段代码为数据加载、预处理和特征提取提供了一个较为完整的框架，适用于基于 PyTorch 的推荐系统模型。
'''
class DataHandler:
	def __init__(self):
		if args.data == 'Beauty':
			predir = './Datasets/Beauty/'
		elif args.data == 'Food':
			predir = './Datasets/Food/'
		elif args.data == 'clothing':
			predir = './Datasets/clothing/'
		elif args.data == 'baby':
			predir = './Datasets/baby/'
		elif args.data == 'sports':
			predir = './Datasets/sports/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'

		self.imagefile = predir + 'image_feat.npy'
		self.textfile = predir + 'text_feat.npy'

	'''
	加载训练/测试集文件
		-用 pickle 加载二进制矩阵文件（用户-项目交互矩阵）。
		-转换为 float32 格式的 稀疏矩阵（COO 格式）。
	'''
	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret


	'''
	邻接矩阵归一化
		-计算度数矩阵 D，然后做归一化
		- GCN 标准做法，使图卷积更稳定。
	'''
	def normalizeAdj(self, mat): 
		degree = np.array(mat.sum(axis=-1)) 	# 节点度数
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0		# 防止除零
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	'''
	构建 PyTorch 稀疏邻接矩阵
		- 将用户-物品矩阵扩展为 二部图邻接矩阵
		- 加上单位矩阵（自连接），做归一化
		- 矩阵转换为稀疏的 PyTorch 张量，并将其转移到 GPU 上。
	'''
	def makeTorchAdj(self, mat):
		a = sp.csr_matrix((args.user, args.user)) # 用户-用户零矩阵
		b = sp.csr_matrix((args.item, args.item)) # 物品-物品零矩阵
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		# make cuda tensor
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()
	
	'''
	加载特征
		- 读取 .npy 文件（图像/文本特征），转换为 GPU 张量。
		- 返回张量和维度。
	'''
	def loadFeatures(self, filename):
		feats = np.load(filename)
		return torch.tensor(feats).float().cuda(), np.shape(feats)[1]

	'''
	主数据加载函数
		- 加载训练和测试矩阵（稀疏矩阵）。
		- 创建训练数据和测试数据的 DataLoader，用于批量加载数据。
		- 加载图像、文本和音频的特征。
		- 生成一个名为 DiffusionData 的数据集实例，用于处理扩散过程的数据。
	'''
	def LoadData(self):
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		self.trnMat = trnMat
		args.user, args.item = trnMat.shape
		self.torchBiAdj = self.makeTorchAdj(trnMat)

		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

		self.image_feats, args.image_feat_dim = self.loadFeatures(self.imagefile)
		self.text_feats, args.text_feat_dim = self.loadFeatures(self.textfile)

		self.diffusionData = DiffusionData(torch.FloatTensor(self.trnMat.A))
		self.diffusionLoader = dataloader.DataLoader(self.diffusionData, batch_size=args.batch, shuffle=True, num_workers=0)

'''
训练数据集
	- 初始化时将训练数据转换为 COO 格式，提取行和列索引，构造字典（DOK）格式的稀疏矩阵。
	- negSampling 方法用于负采样。它从项（item）中随机选取一个负样本（即当前用户没有交互的项目），直到找到一个负样本为止。
'''
class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	# 为每个正样本随机生成一个负样本（用户未交互的物品）。
	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	# 返回 (用户, 正物品, 负物品)
	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

'''
测试数据集
	- 初始化时，构建了一个 CSR 格式的稀疏矩阵（表示用户和项目之间的交互）。
	- 将测试数据按照用户划分，记录每个用户的测试位置（即用户对哪些项目进行了测试）。
	- 返回每个测试样本（用户和该用户的测试数据）。
'''
class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	# 返回 (用户id, 用户的交互向量)
	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])

'''
封装扩散过程的数据（即训练矩阵），扩散数据。
	- 直接把训练矩阵 A 转为行向量，给扩散模型使用。
'''
class DiffusionData(data.Dataset):
	def __init__(self, data):
		self.data = data

	# 返回一行交互记录和对应行索引
	def __getitem__(self, index):
		item = self.data[index]
		return item, index
	
	def __len__(self):
		return len(self.data)