import torch as t
import torch.nn.functional as F

def innerProduct(usrEmbeds, itmEmbeds):
	"""
    计算用户嵌入与物品嵌入的内积（点积），用于衡量用户与物品的匹配分数
    Args:
        usrEmbeds: 用户嵌入张量，形状为 [batch_size, embedding_dim] 或 [num_users, embedding_dim]
        itmEmbeds: 物品嵌入张量，形状与 usrEmbeds 一致（需满足广播条件）
    Returns:
        内积结果，形状为 [batch_size] 或 [num_users]，每个元素表示对应用户-物品对的匹配分数
    """
    # 逐元素相乘后在最后一个维度（嵌入维度）求和，得到内积
	return t.sum(usrEmbeds * itmEmbeds, dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	"""
    计算正负样本对的预测分数差，用于BPR（Bayesian Personalized Ranking）损失计算
    Args:
        ancEmbeds: 锚点嵌入（通常为用户嵌入），形状为 [batch_size, embedding_dim]
        posEmbeds: 正样本物品嵌入，形状为 [batch_size, embedding_dim]
        negEmbeds: 负样本物品嵌入，形状为 [batch_size, embedding_dim]
    Returns:
        正负样本分数差，形状为 [batch_size]，即 (用户-正物品分数) - (用户-负物品分数)
    """
    # 正样本对分数减去负样本对分数，体现"正样本应优于负样本"的排序目标
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

def calcRegLoss(model):
	"""
    计算模型的L2正则化损失，防止模型过拟合
    Args:
        model: 待计算正则化损失的模型（PyTorch的nn.Module实例）
    Returns:
        所有模型参数的L2范数平方和，作为正则化损失
    """
	ret = 0
	# 遍历模型所有可学习参数（权重矩阵、嵌入向量等）
	for W in model.parameters():
		# 累加每个参数的L2范数平方（等价于torch.norm(W, 2)**2）
		ret += W.norm(2).square()
	return ret

def calcReward(bprLossDiff, keepRate):
	"""
    基于BPR损失差异计算边采样的奖励，用于动态调整图中的边保留比例
    Args:
        bprLossDiff: BPR损失的差异值张量，形状为 [num_edges]，衡量每条边对损失的影响
        keepRate: 边的保留比例（例如0.5表示保留50%的边）
    Returns:
        奖励张量，形状与bprLossDiff一致，被选中保留的边对应位置为1.0，否则为0.0
    """
    # 选取损失差异最大的前 (1 - keepRate) 比例的边（即对损失影响最大的边）
	_, posLocs = t.topk(bprLossDiff, int(bprLossDiff.shape[0] * (1 - keepRate)))
	# 初始化奖励张量为0，并将选中的边位置设为1.0
	reward = t.zeros_like(bprLossDiff).cuda()
	reward[posLocs] = 1.0
	return reward

def calcGradNorm(model):
	"""
    计算模型参数梯度的L2范数，用于监控训练过程中的梯度变化（例如梯度爆炸检测）
    Args:
        model: 正在训练的模型（PyTorch的nn.Module实例）
    Returns:
        所有参数梯度的L2范数总和的平方根
    """
	ret = 0
	# 遍历模型所有参数
	for p in model.parameters():
		# 仅考虑存在梯度的参数
		if p.grad is not None:
			# 累加梯度的L2范数平方
			ret += p.grad.data.norm(2).square()
	# 开平方得到总梯度L2范数，并 detach 避免参与梯度计算
	ret = (ret ** 0.5)
	ret.detach()
	return ret

def contrastLoss(embeds1, embeds2, nodes, temp):
	"""
    计算对比学习损失（InfoNCE损失），用于拉近相同实体在不同视角下的嵌入距离
    Args:
        embeds1: 第一视角的嵌入张量，形状为 [num_nodes, embedding_dim]
        embeds2: 第二视角的嵌入张量，形状与embeds1一致
        nodes: 需要计算损失的节点索引，形状为 [batch_size]
        temp: 温度系数，控制相似度分布的陡峭程度
    Returns:
        对比损失值（标量）
    """
	# 对嵌入进行L2归一化，确保相似度计算基于余弦距离
	embeds1 = F.normalize(embeds1, p=2)
	embeds2 = F.normalize(embeds2, p=2)
	# 根据节点索引选取目标节点的嵌入
	pckEmbeds1 = embeds1[nodes] # [batch_size, embedding_dim]
	pckEmbeds2 = embeds2[nodes]
	# 计算正样本对的相似度（同一节点在两视角下的内积），并除以温度系数
	nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp) # [batch_size]
	# 计算每个节点与所有节点的相似度（作为负样本），并求和作为分母
	deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) # [batch_size]
	# 负对数似然的均值作为对比损失（越小表示两视角嵌入越一致）
	return -t.log(nume / deno).mean()
