import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log 	# Utils.TimeLogger: 一个自定义日志工具，用于记录训练过程中的时间信息。
from Params import args 			# Params: 存储脚本中的超参数配置，如学习率、训练轮数等。
from Model import Model, GaussianDiffusion, Denoise
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp 		    
import random 						# random, os: 用于设置随机种子和操作系统环境。
import setproctitle
from scipy.sparse import coo_matrix

'''
实现了一个基于神经网络和扩散模型的推荐系统训练流程，
结合了多模态（如图片、文本）特征，
并使用了对比学习（Contrastive Learning）来增强模型的性能
'''
# Coach —— 训练/评测的总控类
class Coach:
	def __init__(self, handler):
		self.handler = handler 						 # 负责数据与特征的装载

		print('User', args.user, 'Items', args.item)
		print('Number of interactions', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()					  	  # 存放训练/测试各指标随 epoch 演化的记录
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']  # 损失、预处理损失、召回率、归一化折扣累积增益
		for met in mets:
			self.metrics['Train' + met] = list() # 训练期指标曲线
			self.metrics['Test' + met] = list()  # 测试期指标曲线

	'''
	makePrint: 用于格式化输出训练或测试结果，展示每一轮的指标。
	name表示是训练还是测试，ep为当前轮次，reses为包含指标的字典，save表示是否保存结果。
	'''
	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)	# 头部信息：轮次/阶段
		for metric in reses:								# 遍历指标字典
			val = reses[metric]								# 当前指标值
			ret += '%s = %.4f, ' % (metric, val)			# 拼接到打印字符串
			tem = name + metric								# 指标名称（如 TrainLoss）
			if save and tem in self.metrics:				# 若需要保存且指标在记录中
				self.metrics[tem].append(val)				# 将当前指标值添加到对应的列表中
		ret = ret[:-2] + '  '								# 去掉最后的逗号和空格
		return ret

	'''
	run: 训练主流程
         1) 调用 prepareModel 初始化网络/优化器/扩散&去噪模块
         2) 循环 epoch：trainEpoch -> (按周期) testEpoch
         3) 依据 Recall 保存/更新最优结果的统计
	'''
	def run(self):
		self.prepareModel()
		log('Model Prepared')


		recallMax = 0				 # 记录最优 Recall
		ndcgMax = 0		 			 # 记录最优 NDCG
		precisionMax = 0			 # 记录最优 Precision
		bestEpoch = 0		 		 # 记录对应的最优 epoch

		log('Model Initialized')


		for ep in range(0, args.epoch):				# 迭代训练多个 epoch
			tstFlag = (ep % args.tstEpoch == 0)		# 是否在本轮后做一次测试
			reses = self.trainEpoch()				# 训练一个 epoch，返回损失等指标
			log(self.makePrint('Train', ep, reses, tstFlag)) # 打印训练指标，并在需保存时入表
			if tstFlag:
				reses = self.testEpoch()
				# 若本次召回提升，则更新最好结果
				if (reses['Recall'] > recallMax):
					recallMax = reses['Recall']
					ndcgMax = reses['NDCG']
					precisionMax = reses['Precision']
					bestEpoch = ep
				log(self.makePrint('Test', ep, reses, tstFlag))
			print()
		# 训练结束后，打印历史最佳结果
		print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , Precision : ', precisionMax, ' , NDCG', ndcgMax)

	'''
	prepareModel: 构建主模型与优化器；构建扩散模型与两套去噪网络（对应 image/text）
	'''
	def prepareModel(self):
		
		# 主模型：传入已加载的图像/文本特征（detach 防止回传到特征张量）
		self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach()).cuda()
		# 优化器（主模型）
		self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

		# 构建高斯扩散（噪声调度等已在类内 compute）	
		self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).cuda()
		
		# 去噪网络（image 分支）：输入/输出 MLP 结构由 args.dims 指定，末端对齐 item 数量
		out_dims = eval(args.dims) + [args.item]	# 例如 [512, 256, item_num]
		in_dims = out_dims[::-1]				  	# 反向作为编码侧维度
		self.denoise_model_image = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_image = torch.optim.Adam(self.denoise_model_image.parameters(), lr=args.lr, weight_decay=0)

		# 去噪网络（text 分支）：与上面一致，参数独立
		out_dims = eval(args.dims) + [args.item]
		in_dims = out_dims[::-1]
		self.denoise_model_text = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_text = torch.optim.Adam(self.denoise_model_text.parameters(), lr=args.lr, weight_decay=0)

	'''
	normalizeAdj方法
	标准的 GCN 对称归一化 D^{-1/2} A D^{-1/2}
	'''
	def normalizeAdj(self, mat): 
		degree = np.array(mat.sum(axis=-1))				# 每个节点的度（行和）
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1]) # D^{-1/2}
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0			 	# 处理度为 0 的节点
		dInvSqrtMat = sp.diags(dInvSqrt)				# 稀疏对角矩阵
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo() # 归一化并转 COO

	'''
	buildUIMatrix：
	根据 (u_list, i_list, edge_list) 构建 UI 二部图的邻接（带自环 + 归一化），
    并转换成 CUDA 上的稀疏张量，供图传播使用。
	'''
	def buildUIMatrix(self, u_list, i_list, edge_list):
		mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32)

		a = sp.csr_matrix((args.user, args.user)) # U-U 零块
		b = sp.csr_matrix((args.item, args.item)) # I-I 零块
		# 构造二部图邻接：
        # [ 0  UI ]
        # [ UI^T 0 ]
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0     			# 二值化（有边置 1）
		mat = (mat + sp.eye(mat.shape[0])) * 1.0 # 加自环
		mat = self.normalizeAdj(mat)		# GCN 归一化

		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64)) # 边索引
		vals = torch.from_numpy(mat.data.astype(np.float32))				 	# 边权重
		shape = torch.Size(mat.shape)				 							# 张量形状
		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()				# 稀疏张量放到 GPU

	'''
	trainEpoch：训练一个 epoch
      阶段A（扩散去噪预训练/自训练）：
        - 用 denoise_image / denoise_text 在 DiffusionData 上学习重构/一致性
        - 完成后通过 p_sample 生成用户的“伪交互 topK”，重建两张 UI 模态邻接（image/text）
      阶段B（主模型优化）：
        - 前向 (forward_MM) 得到嵌入，计算 BPR + 正则
        - 前向 (forward_cl_MM) 得到多视角嵌入，计算对比损失
        - 反传更新主模型
	'''
	def trainEpoch(self):
		trnLoader = self.handler.trnLoader  		 	# 训练 DataLoader（正样本三元组/负采样）
		trnLoader.dataset.negSampling()					# 刷新本轮的负样本
		epLoss, epRecLoss, epClLoss = 0, 0, 0			# 记录总损失/推荐损失/对比损失（做平均用）
		epDiLoss = 0									# （未直接使用的占位，分别用 image/text 记录）
		epDiLoss_image, epDiLoss_text = 0, 0			# 扩散去噪的两分支平均损失

		steps = trnLoader.dataset.__len__() // args.batch # 训练阶段的步数（以 BPR 训练循环为准）

		diffusionLoader = self.handler.diffusionLoader 	# 用于扩散训练的数据（每行用户交互向量）

		# ---------- 阶段A：训练/更新去噪模型（image/text），得到两张模态 UI 邻接 ----------
		for i, batch in enumerate(diffusionLoader):
			batch_item, batch_index = batch
			batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

			iEmbeds = self.model.getItemEmbeds().detach()	 # 获取当前 item 嵌入（去噪时不更新（冻结））
			uEmbeds = self.model.getUserEmbeds().detach()	 # 获取当前 user 嵌入（去噪时不更新（冻结））（此处未使用）

			image_feats = self.model.getImageFeats().detach()	# 物品图像特征投影到 latdim（冻结）
			text_feats = self.model.getTextFeats().detach()		# 物品文本特征投影到 latdim（冻结）
		
			self.denoise_opt_image.zero_grad()		# 清除 image 去噪网络梯度
			self.denoise_opt_text.zero_grad()		# 清除 text 去噪网络梯度

			# 计算扩散去噪损失 + 一致性约束：
            # diff_loss_*: 重建误差；gc_loss_*: 将 model_output 投影后与 ID 表示对齐的正则
			diff_loss_image, gc_loss_image = self.diffusion_model.training_losses(self.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats)
			diff_loss_text, gc_loss_text = self.diffusion_model.training_losses(self.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats)
			
			# 两分支合成各自的损失（e_loss 为一致性损失权重）
			loss_image = diff_loss_image.mean() + gc_loss_image.mean() * args.e_loss
			loss_text = diff_loss_text.mean() + gc_loss_text.mean() * args.e_loss
			
			epDiLoss_image += loss_image.item()		# 统计本轮 image 分支平均损失
			epDiLoss_text += loss_text.item()		# 统计本轮 text 分支平均损失
			
			loss = loss_image + loss_text			# 合并两分支损失


			loss.backward()							# 反向传播（仅更新去噪网络）
			self.denoise_opt_image.step()			# 更新 image 去噪网络
			self.denoise_opt_text.step()			# 更新 text 去噪网络

			# 打印当前扩散训练进度
			log('Diffusion Step %d/%d' % (i, diffusionLoader.dataset.__len__() // args.batch), save=False, oneline=True) 


		log('')
		log('Start to re-build UI matrix') # 开始基于去噪结果重构两张 UI 模态邻接


		# ---------- 基于去噪采样结果重建 UI 邻接（image/text 两张），用于主模型传播 ----------
		with torch.no_grad():		# 重建阶段不需要梯度

			u_list_image = []		# image 分支用户索引
			i_list_image = []		# image 分支物品索引
			edge_list_image = []	# image 分支边权重（均 1.0）

			u_list_text = []		# text 分支用户索引
			i_list_text = []		# text 分支物品索引
			edge_list_text = []		# text 分支边权重（均 1.0）


			# 再次遍历 DiffusionData，做采样（生成“伪交互”顶级物品集合）
			for _, batch in enumerate(diffusionLoader):
				batch_item, batch_index = batch
				batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

				# image 分支：从去噪网络逐步采样，得到每个用户的重建分数向量
				denoised_batch = self.diffusion_model.p_sample(self.denoise_model_image, batch_item, args.sampling_steps, args.sampling_noise)
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

				# 将 (user, topK_item) 组装为边列表
				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]): 
						u_list_image.append(int(batch_index[i].cpu().numpy()))
						i_list_image.append(int(indices_[i][j].cpu().numpy()))
						edge_list_image.append(1.0)

				# text 分支：同理
				denoised_batch = self.diffusion_model.p_sample(self.denoise_model_text, batch_item, args.sampling_steps, args.sampling_noise)
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]): 
						u_list_text.append(int(batch_index[i].cpu().numpy()))
						i_list_text.append(int(indices_[i][j].cpu().numpy()))
						edge_list_text.append(1.0)


			# 将累计的三元组转为 numpy 数组，并调用 buildUIMatrix 构建两张邻接
            # image UI 邻接
			u_list_image = np.array(u_list_image)
			i_list_image = np.array(i_list_image)
			edge_list_image = np.array(edge_list_image)
			self.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
			self.image_UI_matrix = self.model.edgeDropper(self.image_UI_matrix)

			# text UI 邻接
			u_list_text = np.array(u_list_text)
			i_list_text = np.array(i_list_text)
			edge_list_text = np.array(edge_list_text)
			self.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
			self.text_UI_matrix = self.model.edgeDropper(self.text_UI_matrix)

		log('UI matrix built!')


		# ---------- 阶段B：主模型训练（BPR + 正则 + 对比学习） ----------
		for i, tem in enumerate(trnLoader):		# 遍历训练三元组 (user, pos, neg)
			ancs, poss, negs = tem				# anchors/users；positive items；negative items
			ancs = ancs.long().cuda()			# 将用户索引转为 CUDA 张量
			poss = poss.long().cuda()			# 将正物品索引转为 CUDA 张量
			negs = negs.long().cuda()			# 将负样本物品索引转为 CUDA 张量

			self.opt.zero_grad()			# 清除主模型梯度

			# 使用主图 + 两张模态邻接做多模态传播，得到最终嵌入
			usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
			# 取出当前 batch 的 user/pos/neg 嵌入
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			negEmbeds = itmEmbeds[negs]
			# BPR 打分差：s(u,pos) - s(u,neg)，pairPredict 通常是内积或相似度
			scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
			# BPR 损失：-log σ(diff)，并对 batch 求均值
			bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
			# L2 正则（用户/物品嵌入），权重 args.reg
			regLoss = self.model.reg_loss() * args.reg
			# 总损失先累加 BPR+Reg
			loss = bprLoss + regLoss
			
			epRecLoss += bprLoss.item()	# 记录推荐损失
			epLoss += loss.item()		# 记录总损失（后续还会加 CL）


			# 多视角对比：用 image_adj/text_adj 初始扩散得到两套视角嵌入，再在主图上 GCN
			usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
			

			# 视角内对比（img vs txt）：用户与物品各做一次对比损失（温度 args.temp）
			clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg

			# 与融合视角（forward_MM 输出）做对比：提高多视角一致性
			clLoss1 = (contrastLoss(usrEmbeds, usrEmbeds1, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds1, poss, args.temp)) * args.ssl_reg
			clLoss2 = (contrastLoss(usrEmbeds, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds2, poss, args.temp)) * args.ssl_reg

			clLossAll = clLoss1 + clLoss2 	# 组合对比项

			if args.cl_method == 1:
				clLoss = clLossAll			# 若指定仅用组合对比

			loss += clLoss					# 总损失加入对比项

			epClLoss += clLoss.item()		# 记录对比损失

			loss.backward()					# 反向梯度
			self.opt.step()					# 更新主模型参数

			log('Step %d/%d: bpr : %.3f ; reg : %.3f ; cl : %.3f ' % (
				i, 
				steps,
				bprLoss.item(),
        		regLoss.item(),
				clLoss.item()
				), save=False, oneline=True)

		# 组装返回本 epoch 的均值统计
		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['BPR Loss'] = epRecLoss / steps
		ret['CL loss'] = epClLoss / steps
		ret['Di image loss'] = epDiLoss_image / (diffusionLoader.dataset.__len__() // args.batch)
		ret['Di text loss'] = epDiLoss_text / (diffusionLoader.dataset.__len__() // args.batch)
		return ret

	
	'''
	testEpoch: 在测试集上评测一个 epoch
               - 前向得到用户/物品嵌入
               - 逐 batch 计算 Top-K 推荐，评估 Recall/NDCG/Precision
	'''
	def testEpoch(self):
		tstLoader = self.handler.tstLoader			# 测试 DataLoader（按用户划分）
		epRecall, epNdcg, epPrecision = [0] * 3		# 累计指标
		i = 0
		num = tstLoader.dataset.__len__()			# 测试用户数
		steps = num // args.tstBat

		# 用融合视角的前向得到整体用户/物品表示（与训练阶段一致）
		usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)

		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().cuda()		# 用户索引
			trnMask = trnMask.cuda()	# 用户在训练集的交互掩码（已交互置 1）

			# 计算所有物品的预测分数：U @ I^T，屏蔽训练中出现过的物品
			allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			_, topLocs = torch.topk(allPreds, args.topk)
			recall, ndcg, precision = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
			epRecall += recall
			epNdcg += ndcg
			epPrecision += precision
			log('Steps %d/%d: recall = %.2f, ndcg = %.2f , precision = %.2f   ' % (i, steps, recall, ndcg, precision), save=False, oneline=True)
		# 汇总平均指标
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		ret['Precision'] = epPrecision / num
		return ret

	
	'''
	calcRes: 计算一个 batch 用户的三类指标
             - topLocs: 预测 Top-K 的物品索引数组（shape: [B, K]）
             - tstLocs: 测试集中每个用户的真实点击物品列表
             - batIds : 当前 batch 的用户 id 张量
	'''
	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)   	# 校验 batch 对齐
		allRecall = allNdcg = allPrecision = 0		# 累计器
		for i in range(len(batIds)):				# 遍历 batch 内每个用户
			temTopLocs = list(topLocs[i])			# 当前用户的 Top-K 推荐列表
			temTstLocs = tstLocs[batIds[i]]			# 当前用户在测试集的真实物品列表
			tstNum = len(temTstLocs)				# 真实物品个数
			# 计算理想 DCG（IDCG）：位置从 0..K-1
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			recall = dcg = precision = 0			# 单用户三指标
			for val in temTstLocs:
				if val in temTopLocs:				# 若命中 Top-K
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2)) # 按排名衰减累加
					precision += 1
			recall = recall / tstNum				# Recall@K
			ndcg = dcg / maxDcg						# NDCG@K：实际 DCG 除以理想 DCG
			precision = precision / args.topk		# Precision@K：命中物品数除以 K
			allRecall += recall
			allNdcg += ndcg
			allPrecision += precision
		return allRecall, allNdcg, allPrecision		# 返回 batch 累计（上层会再做平均）


# 随机性控制函数
def seed_it(seed):
	random.seed(seed)					 		# Python 随机数种子
	os.environ["PYTHONSEED"] = str(seed)		# 记录环境变量（可选）
	np.random.seed(seed)						# NumPy 随机数种子
	torch.cuda.manual_seed(seed)				# CUDA 随机（当前设备）
	torch.cuda.manual_seed_all(seed)			# CUDA 随机（所有设备）
	torch.backends.cudnn.deterministic = True	# cuDNN 结果可复现（可能稍慢）
	torch.backends.cudnn.benchmark = True 		# 启用卷积调优（加速；与 deterministic 取舍）
	torch.backends.cudnn.enabled = True			# 启用 cuDNN（默认开启）
	torch.manual_seed(seed)						# PyTorch CPU 随机数种子	

if __name__ == '__main__':
	seed_it(args.seed)			# 设定随机种子

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # 指定可见 GPU（如 "0" 或 "0,1"）
	
	logger.saveDefault = True	# TimeLogger 的默认保存开关
	
	log('Start')
	handler = DataHandler() 	# 实例化数据处理器
	handler.LoadData()			# 加载矩阵/特征/构造 DataLoader
	log('Load Data')	

	coach = Coach(handler)		# 创建训练总控
	coach.run()					# 启动训练流程		

	
