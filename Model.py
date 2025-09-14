import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
import math
from Utils.Utils import *

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
'''
结合了图神经网络（GCN）、扩散模型（Diffusion）、多模态特征融合（图像、文本）以及对比学习技术
'''
class Model(nn.Module):
	def __init__(self, image_embedding, text_embedding):
		super(Model, self).__init__()
		# 调用父类 nn.Module 的初始化方法，确保 PyTorch 模块正常构建。

    	# 用户和物品的基础嵌入向量参数，形状分别是 [user数量, 潜在维度] 和 [item数量, 潜在维度]。
    	# nn.Parameter：这些向量会在训练中作为模型参数更新。
		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))

		# 定义多层图卷积层（GCNLayer），用 nn.Sequential 顺序堆叠 args.gnn_layer 层。
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)]) 

		# 边采样器，用于在训练时对邻接矩阵随机丢弃部分边（DropEdge），增强模型泛化能力。
		self.edgeDropper = SpAdjDropEdge(args.keepRate)

	
		'''
		将原始模态特征（图像 / 文本 ）映射到相同的潜在维度 latdim。
    	- args.trans 控制使用哪种变换方式：
        1 → 使用线性层 (nn.Linear)
        0 → 使用参数矩阵 (nn.Parameter)
        其他 → 图像用矩阵，文本用线性层
		'''
		if args.trans == 1:
			self.image_trans = nn.Linear(args.image_feat_dim, args.latdim)
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
		elif args.trans == 0:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Parameter(init(torch.empty(size=(args.text_feat_dim, args.latdim))))
		else:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)

		# 保存输入进来的预训练图像和文本 embedding（固定特征）
		self.image_embedding = image_embedding
		self.text_embedding = text_embedding

		'''
		模态权重参数（融合图像和文本特征时使用）。
    	初始值为 [0.5, 0.5]，通过 softmax 保证归一化后权重和为 1。
		'''

		self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
		self.softmax = nn.Softmax(dim=0)

		# Dropout：防止过拟合
		self.dropout = nn.Dropout(p=0.1)

		# LeakyReLU 激活函数（斜率 0.2），用于非线性变换
		self.leakyrelu = nn.LeakyReLU(0.2)

	# 返回当前训练得到的物品嵌入矩阵 iEmbeds。
    # 形状是 [item数, 潜在维度]。
    # 在推荐任务中，每一行表示一个物品在潜在空间中的表示向量。			
	def getItemEmbeds(self):
		return self.iEmbeds
	

	# 返回当前训练得到的用户嵌入矩阵 uEmbeds。
    # 形状是 [user数, 潜在维度]。
    # 在推荐任务中，每一行表示一个用户在潜在空间中的表示向量。
	def getUserEmbeds(self):
		return self.uEmbeds
	
	'''
	def getImageFeats(self), getTextFeats(self)
	这两个方法用于返回**转换后的模态特征向量**。
	- 原始输入是预训练好的 image_embedding / text_embedding（静态特征）。
	- 根据 args.trans 的不同值，使用不同方式做维度映射。
	- 如果使用参数矩阵转换（torch.mm），还会通过 LeakyReLU 做非线性激活。
	'''
	def getImageFeats(self):
		if args.trans == 0 or args.trans == 2:
			# 如果 args.trans=0 或 2，使用参数矩阵进行线性变换：
			# image_embedding @ image_trans
			# 然后经过 LeakyReLU 增强非线性表达。
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			return image_feats
		else:
			# 如果 args.trans=1，则直接用 nn.Linear 做映射。
			return self.image_trans(self.image_embedding)
	
	def getTextFeats(self):
		if args.trans == 0:
			# 如果 args.trans=0，使用参数矩阵进行线性变换：
			# text_embedding @ text_trans
			# 再经过 LeakyReLU 激活。
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
			return text_feats
		else:
			# 如果 args.trans=1 或 2，则直接用 nn.Linear 做映射。
			return self.text_trans(self.text_embedding)

	'''
	正向传播 forward_MM
	'''
	def forward_MM(self, adj, image_adj, text_adj):
		# 1) 模态特征映射到统一维度（latdim）
		#    - args.trans=0/2：使用参数矩阵 + LeakyReLU
		#    - args.trans=1  ：使用 nn.Linear
		image_feats = self.getImageFeats() # [item, D]
		text_feats = self.getTextFeats()   # [item, D]

		# 2) 计算模态融合权重（两模态：图像/文本）
		weight = self.softmax(self.modal_weight) # [2]，weight[0] 对应图像，weight[1] 对应文本


		# 3) 基于 image_adj 做一次“图像模态通道”的图传播
    	#    先把当前节点表示拼起来：顺序是 [用户U; 物品I]，维度 [N, D]
		embedsImageAdj = torch.concat([self.uEmbeds, self.iEmbeds])
		#    使用稀疏乘法 spmm 沿 image_adj 传播
		embedsImageAdj = torch.spmm(image_adj, embedsImageAdj)

		# 4) 构造“图像模态表征”：用户用 uEmbeds，物品用 归一化后的图像特征
		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])

		# 5) 在交互图 adj 上做两跳式传播（本行 & 下几行）
    	#    第 1 跳：把“用户表示 + 物品原始 iEmbeds”拼起来，通过 adj 传播到所有节点
		embedsImage_ = torch.concat([embedsImage[:args.user], self.iEmbeds]) # [N, D]
		embedsImage_ = torch.spmm(adj, embedsImage_)    					# [N, D]
		embedsImage += embedsImage_											# 残差叠加
		
		# 6) 文本模态通道：与上面图像通道类似
		embedsTextAdj = torch.concat([self.uEmbeds, self.iEmbeds]) # [N, D]
		embedsTextAdj = torch.spmm(text_adj, embedsTextAdj) 	# [N, D]

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)]) # [N, D]
		embedsText = torch.spmm(adj, embedsText)						# 第 1 跳

		embedsText_ = torch.concat([embedsText[:args.user], self.iEmbeds]) # [N, D]
		embedsText_ = torch.spmm(adj, embedsText_)						# 第 2 跳
		embedsText += embedsText_										# 残差叠加

		# 7) 将“模态融合边”的传播结果（embedsImageAdj/embedsTextAdj）以 λ 加权注入
		embedsImage += args.ris_adj_lambda * embedsImageAdj
		embedsText += args.ris_adj_lambda * embedsTextAdj

		# 8) 两模态融合：按可学习权重做凸组合
		embedsModal = weight[0] * embedsImage + weight[1] * embedsText # [N, D]


		# 9) 多层 GCN：用交互图 adj 对 embedsModal 进行 L 层图卷积（带层内残差累计）
		embeds = embedsModal
		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1]) # 单层 GCN: spmm + 非线性/归一化（由 GCNLayer 内部定义）
			embedsLst.append(embeds)
		embeds = sum(embedsLst)             # 层间残差：把每层输出相加（有助稳定训练 & 信息聚合）

		# 10) 最终残差正则：再把规范化后的模态融合表征以 λ 注入
		embeds = embeds + args.ris_lambda * F.normalize(embedsModal) # [N, D]

		# 11) 切分回 用户/物品
		return embeds[:args.user], embeds[args.user:] # u_final: [U,D], i_final: [I,D]

	'''
	对比学习模块 forward_cl_MM
	返回图像、文本三个视角下分别处理过的用户与物品嵌入。用于对比学习（contrastive learning）。
	'''
	def forward_cl_MM(self, adj, image_adj, text_adj):
		# 1) 模态特征转换到统一潜在维度 D（args.latdim）
		#    - args.trans=0：参数矩阵 + LeakyReLU
		#    - args.trans=1：nn.Linear 
		#    - 其他：图像用参数矩阵+激活，文本用 nn.Linear
		if args.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		elif args.trans == 1:
			image_feats = self.image_trans(self.image_embedding)
			text_feats = self.text_trans(self.text_embedding)
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.text_trans(self.text_embedding)

		# 2) 构造两种“模态视角”的初始节点表示，并用各自的模态图先做一次传播（spmm）
		#    约定节点顺序仍是 [用户; 物品]
		#    - 图像视角：用户用 uEmbeds，物品用 归一化后的 image_feats
		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
		embedsImage = torch.spmm(image_adj, embedsImage)			# 在图像模态的图上扩散

		#    - 文本视角：用户用 uEmbeds，物品用 归一化后的 text_feats
		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		embedsText = torch.spmm(text_adj, embedsText)				# 在文本模态的图上扩散

 		# 3) 再把两种视角的表示各自送入同一套多层 GCN（用主交互图 adj）做深层传播
    	#    这里的做法：层间残差（把每层输出累加），有助于信息聚合与梯度稳定
		embeds1 = embedsImage 			# 图像视角起点
		embedsLst1 = [embeds1]
		for gcn in self.gcnLayers:
			embeds1 = gcn(adj, embedsLst1[-1]) # 单层 GCN: spmm + 归一化/非线性（由 GCNLayer 内部实现）
			embedsLst1.append(embeds1)
		embeds1 = sum(embedsLst1)		    # 层间残差聚合 => 图像视角最终表示 [N, D]	

		embeds2 = embedsText			# 文本视角起点 
		embedsLst2 = [embeds2]
		for gcn in self.gcnLayers:
			embeds2 = gcn(adj, embedsLst2[-1])
			embedsLst2.append(embeds2)
		embeds2 = sum(embedsLst2) 		# 文本视角最终表示 [N, D]


		# 4) 切分出用户/物品嵌入（按 [用户; 物品] 的拼接顺序）
		return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:]


	'''
	def reg_loss(self)正则化损失
	对用户和物品嵌入做 L2 正则，防止过拟合。
	'''
	def reg_loss(self):
		ret = 0
		ret += self.uEmbeds.norm(2).square()
		ret += self.iEmbeds.norm(2).square()
		return ret

'''
GCN 图卷积层 GCNLayer “极简版的 GCN 图卷积层”
	稀疏矩阵乘法：adj @ embeds
	简洁但高效，用于图结构信息传播。
'''
class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()
        self.act = nn.LeakyReLU(0.2)
        # 增加图注意力权重参数
        self.att_weight = nn.Parameter(init(torch.empty(args.latdim, 1)))
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, adj, embeds):
        # 标准GCN传播
        embeds = torch.spmm(adj, embeds)
        # 计算注意力权重
        att_scores = torch.matmul(embeds, self.att_weight).squeeze()
        att_scores = F.softmax(att_scores, dim=0).unsqueeze(1)
        # 应用注意力权重
        embeds = embeds * att_scores
        # 非线性变换和dropout
        embeds = self.act(embeds)
        return self.dropout(embeds)

'''
边丢弃模块 SpAdjDropEdge
	使用随机掩码 mask 丢弃一部分邻接矩阵中的边，提升模型鲁棒性。
	丢弃后边权值需除以 keepRate 保持期望不变。
'''
class SpAdjDropEdge(nn.Module):
	def __init__(self, keepRate):
		super(SpAdjDropEdge, self).__init__()
		self.keepRate = keepRate

	def forward(self, adj):
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

		newVals = vals[mask] / self.keepRate
		newIdxs = idxs[:, mask]

		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)

'''
去噪模型 Denoise
构造函数
	1、in_dims, out_dims: 网络输入输出结构（多层感知机）。
	2、emb_layer: 时间嵌入（sin-cos 编码后线性映射）。
	3、ropout: 防止过拟合。
	所有权重使用高斯初始化（std = sqrt(2 / (in + out))）。
'''
class Denoise(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
		super(Denoise, self).__init__()
		self.in_dims = in_dims 		 # 输入层和隐藏层的维度列表
		self.out_dims = out_dims	 # 输出层和隐藏层的维度列表
		self.time_emb_dim = emb_size # 时间嵌入维度
		self.norm = norm			 # 是否对输入做归一化

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim) # 时间嵌入的线性映射层

		# 输入层维度要额外拼接时间嵌入维度
		in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
		out_dims_temp = self.out_dims

		# 输入侧网络（编码层）：多层感知机
		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		# 输出侧网络（解码层）：多层感知机
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

		self.drop = nn.Dropout(dropout) # Dropout 防止过拟合
		self.init_weights()				# 权重初始化

	# 权重初始化
	def init_weights(self):
		# 编码层权重初始化
		for layer in self.in_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1])) # He 初始化：std = sqrt(2 / (fan_in + fan_out))
			layer.weight.data.normal_(0.0, std)		 # 正态分布初始化权重
			layer.bias.data.normal_(0.0, 0.001)		 # 偏置初始化为接近 0
		
		# 解码层权重初始化
		for layer in self.out_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		# 时间嵌入层权重初始化
		size = self.emb_layer.weight.size()
		std = np.sqrt(2.0 / (size[0] + size[1]))
		self.emb_layer.weight.data.normal_(0.0, std)
		self.emb_layer.bias.data.normal_(0.0, 0.001)

	'''
	前向传播
		1、时间嵌入构建（基于 sin/cos 函数）；
		2、拼接到原始输入 x；
		3、通过一组编码层（in_layers）和解码层（out_layers）；
		4、输出重构结果。
	'''
	def forward(self, x, timesteps, mess_dropout=True):
		# 1. 构建时间嵌入（sin-cos 位置编码）
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		temp = timesteps[:, None].float() * freqs[None] 				 # [batch, dim/2]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:										 # 如果时间嵌入维度是奇数，补齐最后一维为 0			
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		# 2. 时间嵌入通过线性层映射
		emb = self.emb_layer(time_emb)
		# 3. 对输入特征做归一化和 dropout（可选）
		if self.norm:
			x = F.normalize(x)
		if mess_dropout:
			x = self.drop(x)
		# 4. 拼接输入特征和时间嵌入
		h = torch.cat([x, emb], dim=-1)

		# 5. 输入侧 MLP（编码）
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)
		
		# 6. 输出侧 MLP（解码）
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:  # 最后一层不加激活
				h = torch.tanh(h)

		return h

'''
扩散模型 GaussianDiffusion
	初始化 beta 序列（添加噪声的强度）；
	预计算 alphas_cumprod 等一系列用于扩散反扩散的中间变量。
'''
class GaussianDiffusion(nn.Module):
	# 构造函数
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()

		self.noise_scale = noise_scale	# 噪声缩放因子
		self.noise_min = noise_min		# 最小噪声值
		self.noise_max = noise_max		# 最大噪声值
		self.steps = steps				# 扩散步数 T

		if noise_scale != 0:
			# 生成 betas（每一步的噪声强度）
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			if beta_fixed:			   # 修正第一个 beta，避免过小
				self.betas[0] = 0.0001

			# 预计算扩散过程所需变量
			self.calculate_for_diffusion()

	# 生成噪声调度表 
	# 作用：生成 β 序列，控制每个扩散步骤的噪声量。
	def get_betas(self):
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max 
		variance = np.linspace(start, end, self.steps, dtype=np.float64) # 噪声方差线性递增
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			# 保证 β_t ∈ [0,1)，避免不稳定
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas) 

	# 预计算参数
	# 作用：提前算好各种扩散/反扩散需要的系数，提高采样速度。
	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

		# 常用辅助变量
		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		# 后验分布参数（q(x_{t-1}|x_t, x_0)）
		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	# 反向采样（生成过程）
	# 作用：逐步去噪，把随机噪声还原成数据。
	def p_sample(self, model, x_start, steps, sampling_noise=False):
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t) 	# 给原始数据加噪声 → 得到 x_t
		
		indices = list(range(self.steps))[::-1] # 从 T 到 0 逐步还原

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
			if sampling_noise: # 是否添加随机性
				noise = torch.randn_like(x_t)
				nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
				x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
			else:
				x_t = model_mean
		return x_t


	# 正向扩散（加噪过程）
	# 作用：把原始样本 x_start 加噪，得到第 t 步的 x_t。
	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

	# 作用：把标量参数（如 α̅_t）扩展成与输入张量同形状，方便做逐元素运算。
	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)

	# 估计后验均值和方差
	# 作用：根据模型预测结果，计算反扩散分布参数。
	def p_mean_variance(self, model, x, t):
		model_output = model(x, t, False) # 预测 x_0 或噪声

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		
		return model_mean, model_log_variance

	# 计算训练损失
	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
		batch_size = x_start.size(0)

		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda() # 随机采样时间步
		noise = torch.randn_like(x_start)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise) # 加噪
		else:
			x_t = x_start

		model_output = model(x_t, ts)  # 模型预测（去噪）

		# MSE 损失
		mse = self.mean_flat((x_start - model_output) ** 2)

		# SNR 权重
		weight = self.SNR(ts - 1) - self.SNR(ts)
		weight = torch.where((ts == 0), 1.0, weight)

		diff_loss = weight * mse 	# 扩散损失（重建误差）
		# 用户嵌入约束（保持一致性）
		usr_model_embeds = torch.mm(model_output, model_feats)
		usr_id_embeds = torch.mm(x_start, itmEmbeds)
		gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

		return diff_loss, gc_loss
		
	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape)))) # 除 batch 维外取均值
	
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t]) # 信噪比

