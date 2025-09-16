

---

# Preference Rebuilder (PR)

**Preference Rebuilder (PR)** 是一个面向推荐系统的 **噪声感知偏好重建框架**。它通过“扰动—重建”的训练范式，从稀疏/带噪的用户交互中恢复出完整的偏好分数向量，并基于多模态特征进行对齐校准，从而生成高质量的候选集与伪交互边，提升推荐效果。

---

## 🔑 特点

* **噪声感知重建**
  在训练阶段引入可控强度的扰动，迫使模型学习在不同质量输入下的鲁棒复原能力。

* **双空间校准**
  重建分数在 **ID 空间** 和 **内容空间** 同时保持一致，避免偏科。

* **候选与伪边生成**
  PR 输出的致密偏好分数可直接取 Top-K，作为候选物品或伪交互边。

* **图结构增强**
  将伪边注入用户—物品图，与原始交互共同传播，提升嵌入质量与召回性能。

---

## 🏗️ 框架概览

```
用户交互向量 r_u  ─┐
                    │扰动引擎 (Corruption Engine)
                    ▼
             扰动后的输入  r̃_u^s
                    │ + stage s
                    ▼
               重建网络 (RebuildNet)
                    ▼
          重建偏好分数  p̂_u ∈ R^I
                    │
     ┌──────────────┴──────────────┐
     │                             │
内容投影校准                 ID 投影校准
     │                             │
     └───── 一致性约束 ────────────┘
                    │
            链路生成 (Top-K)
                    ▼
       伪交互边/候选集 → 图传播融合
```

---

## ⚙️ 安装与依赖

```bash
git clone https://github.com/yourname/preference-rebuilder.git
cd preference-rebuilder
pip install -r requirements.txt
```

主要依赖：

* Python ≥ 3.8
* PyTorch ≥ 1.10
* NumPy, SciPy
* tqdm 等工具库

---

## 🚀 快速开始

### 1. 数据准备

* 将用户—物品交互存为稀疏矩阵 (CSR/COO)。
* 准备物品的多模态特征（如图像/文本 embedding）。

### 2. 训练 PR

```bash
python Main.py --dataset yourdata --latdim 128 --steps 5 --rebuild_k 5
```

主要参数：

* `--steps` : 扰动—重建的 stage 数量
* `--rebuild_k` : 每个用户生成的伪边数量
* `--e_loss` : 校准损失权重
* `--ssl_reg` : 对比学习正则系数

### 3. 推理与评估

训练完成后，PR 会输出：

* **重建的候选 Top-K**
* **伪交互边图**（可与原始图整合）
* Recall / NDCG / Precision 等评估指标

---

## 📊 实验结果

| 数据集          | Recall\@20 | NDCG\@20 | 覆盖率提升 |
| ------------ | ---------- | -------- | ----- |
| MovieLens-1M | 0.xx       | 0.xx     | +x%   |
| Yelp         | 0.xx       | 0.xx     | +x%   |

*(示例表格，请替换为你的实验结果)*

---

## 📌 应用场景

* 冷启动推荐：用户交互稀疏，PR 可补全偏好。
* 多模态融合：内容特征与交互信号统一建模。
* 图结构增强：为图神经网络提供更致密的邻接。

---

## 🔮 未来方向

* 引入更多扰动策略（如曝光偏置建模）。
* 尝试迭代式多阶段重建。
* 在大规模工业数据集上验证鲁棒性与效率。

---

## 🤝 引用 / 致谢

如果你在研究中使用了 **Preference Rebuilder (PR)**，请引用本项目：

```bibtex
@misc{preference_rebuilder,
  title={Preference Rebuilder: A Noise-aware Preference Reconstruction Framework for Recommendation},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourname/preference-rebuilder}}
}
```

---

要不要我帮你再写一版 **中文 README**（更偏向报告/内部分享），还是保持这种英文开源模板风格呢？
