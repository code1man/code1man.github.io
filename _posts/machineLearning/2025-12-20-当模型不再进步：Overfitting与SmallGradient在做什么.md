---
layout:     post
title:      当模型不再进步：Overfitting 与 Small Gradient 在做什么
date:       2025-12-20
subtitle:   深度神经网络的两大训练难题：过拟合与梯度消失
categories: 机器学习
author:     蓝
header-img: img/machineLearning/教士菲比.jpg
catalog: true
tags: 
  - 机器学习
---

# 目录
- [目录](#目录)
- [Loss的四种结局与训练诊断决策树](#loss的四种结局与训练诊断决策树)
  - [1. Training loss 很大（large）](#1-training-loss-很大large)
    - [1.1 Small Gradient 出现时的解决方法](#11-small-gradient-出现时的解决方法)
      - [1.1.1 回到 Batch：Batch、Epoch、Shuffle 基础](#111-回到-batchbatchepochshuffle-基础)
      - [1.1.2 Small Batch v.s. Large Batch](#112-small-batch-vs-large-batch)
      - [1.1.3 Small Batch Size 的优势与解释](#113-small-batch-size-的优势与解释)
      - [1.1.4 Momentum 算法](#114-momentum-算法)
    - [1.2 自适应学习率方法](#12-自适应学习率方法)
      - [1.2.1 1/t 衰减：学习率随训练进程逐步减小。](#121-1t-衰减学习率随训练进程逐步减小)
      - [1.2.2 Adagrad](#122-adagrad)
      - [1.2.3 随机梯度下降（SGD）：每次用一个 batch 更新参数，带来“噪声”有助于泛化。](#123-随机梯度下降sgd每次用一个-batch-更新参数带来噪声有助于泛化)
      - [1.2.4. 特征缩放（Feature Scaling）：将所有特征缩放到均值为 0、方差为 1，避免某些维度主导梯度，提升收敛速度。](#124-特征缩放feature-scaling将所有特征缩放到均值为-0方差为-1避免某些维度主导梯度提升收敛速度)
      - [1.2.5 RMSProp](#125-rmsprop)
      - [1.2.6 Adam](#126-adam)
      - [1.2.7 Warmup](#127-warmup)
  - [2. Training loss 很小](#2-training-loss-很小)
    - [2.1 Overfitting（过拟合）](#21-overfitting过拟合)
      - [2.1.1 增强训练数据](#211-增强训练数据)
      - [2.1.2 降低模型复杂度](#212-降低模型复杂度)
      - [2.1.3 正则化（Regularization）](#213-正则化regularization)
      - [2.1.4 验证集的使用（Validation Set）](#214-验证集的使用validation-set)
      - [2.1.5 N-fold 交叉验证（N-fold Cross Validation）](#215-n-fold-交叉验证n-fold-cross-validation)
    - [2️.2 Mismatch（数据分布不匹配）](#2️2-mismatch数据分布不匹配)
  - [3. 总结](#3-总结)
    - [3.1 具体区分情况](#31-具体区分情况)


# Loss的四种结局与训练诊断决策树

> PS: 前情提要：
    **<font color=yellowgreen>1. 梯度爆发/消失原因</font>**
    **梯度爆发**：网络层数深、权重初始化过大，导致梯度在反向传播时指数级增长。
    **梯度消失**：激活函数（如 sigmoid、tanh）导致梯度在多层传播后趋近于 0，参数更新缓慢甚至停滞。本质是<font color=camel>矩阵乘积的谱性质导致的指数衰减</font>

在机器学习训练过程中，Loss（损失函数）的变化揭示了模型学习的状态。李宏毅老师用一张训练诊断决策树（General Guide）总结了Loss的四种典型结局，并给出了对应的分析与对策：

![训练诊断决策树](/img/machineLearning/Loss诊断/训练诊断决策树.png)

## 1. Training loss 很大（large）

**说明：**  模型连训练集都拟合不好，这通常不是 overfitting，而是：

- **原因一：Model Bias（模型偏差大）**  
  - 模型太简单，表达能力不够，学不到数据里的规律。
  - **对策：** 增加参数、加深网络、引入非线性（make your model complex）。

- **原因二：Optimization 问题**  

  - **Optimization（优化）** 指的是通过算法（如梯度下降）调整模型参数，使损失函数最小化的过程。
  - 可能原因：learning rate 不合适、small gradient/vanishing gradient、初始化问题等。

  > **如何区分 Model Bias 和 Optimization 问题？**  
      - 如果模型结构本身太简单，训练集表现就很差，通常是 Model Bias（模型偏差）导致的。
      - <font color=green>如果模型理论上足够复杂</font>，但训练集 loss 依然很大，通常是 Optimization 问题（优化没做好），比如学习率、梯度消失/爆炸、初始化等。

### 1.1 Small Gradient 出现时的解决方法

#### 1.1.1 回到 Batch：Batch、Epoch、Shuffle 基础

- **Batch（批次）**：一次前向/反向传播所用的数据子集。  
- **Epoch（轮次）**：所有训练数据被完整训练一次。  
- **Shuffle（打乱）**：每个 epoch 前将数据顺序随机打乱，避免模型记忆数据顺序，提高泛化能力。  
- **示例代码（以 TensorFlow 为例）：**

  ```python
  dataset = tf.data.Dataset.from_tensor_slices(data)
  dataset = dataset.shuffle(buffer_size=len(data))  # buffer_size 越大打乱越彻底
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(num_epochs)

  ```

  > <font color=chocolate>注意：`batch()` 应在 `shuffle()` 之后调用，否则 batch 内部不会被打乱。</font>

#### 1.1.2 Small Batch v.s. Large Batch

- **Small Batch（小批量）**：每次更新参数用的数据量小，更新“噪声”大，训练更“跳跃”。
- **Large Batch（大批量）**：每次更新用的数据量大，更新更“平滑”，但可能陷入鞍点或局部最小值。
- **效率与准确率**：不考虑 GPU 并行时，小 batch 更新快但一个 epoch 慢；大 batch 一次慢但 epoch 快。实际用大 batch 结合 GPU 并行，速度未必慢。
- **为什么用 batch？**  
  - 计算效率：批量计算可用向量化加速。
  - 泛化能力：小 batch 的“噪声”有助于跳出局部最优。

#### 1.1.3 Small Batch Size 的优势与解释

- **现象**：小 batch size 训练效果往往更好（Smaller batch size has better performance）。
- **原因一（Noisy Update）**：小 batch 的“噪声”让参数更新轨迹多样，能跳出鞍点/局部最小值，不易被卡住。  
- **原因二（平坦极小值）**：深度学习更偏好“平坦”的极小值（flat minima），小 batch 更容易找到这些区域。平坦极小值对泛化更友好，因为参数微小扰动不会导致性能大幅下降。  
- **可拓展推荐论文**：  [`Keskar et al., 2017, "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"`](https://arxiv.org/abs/1609.04836)指出小 batch 更容易找到平坦极小值，泛化更好。

#### 1.1.4 Momentum 算法

- **原理**：参数更新不仅依赖当前梯度，还考虑之前的“动量”，类似物理中的惯性。
- **公式**：  
  $$
  v_{t+1} = \gamma v_t + \eta \nabla L(\theta_t) \\
  \theta_{t+1} = \theta_t - v_{t+1}
  $$
  其中 $\gamma$ 为动量系数，$\eta$ 为学习率。
- **代码示例（PyTorch）**：

  ```python
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

  ```
- **总结**：小 batch size 和 momentum 有助于跳出鞍点/局部最优。


> PS： **Training stuck ≠ Small Gradient**：训练停滞不一定是梯度消失，可能是不同参数需要不同学习率。

### 1.2 自适应学习率方法
#### 1.2.1 1/t 衰减：学习率随训练进程逐步减小。
#### 1.2.2 Adagrad
- **原理**：对每个参数自适应调整学习率，历史梯度大则步长小，反之亦然。
- **公式**：  
    $$
    \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla L(\theta_t)
    $$
    其中 $G_t$ 是历史梯度平方和。
- **代码示例（PyTorch）**：

    ```python
      optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)

    ```
- **本质解释**：用一阶导数的平方和近似二阶导数，自动调整步长。更大的梯度不一定意味着远离最小值，因为不同参数的梯度尺度不同，不能跨参数直接比较。
- **参考论文**：[Duchi et al., 2011, "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"](https://jmlr.org/papers/v12/duchi11a.html)
#### 1.2.3 随机梯度下降（SGD）：每次用一个 batch 更新参数，带来“噪声”有助于泛化。
#### 1.2.4. 特征缩放（Feature Scaling）：将所有特征缩放到均值为 0、方差为 1，避免某些维度主导梯度，提升收敛速度。
#### 1.2.5 RMSProp
- **原因**：SGD 和 Adagrad 在训练深度网络时，可能因学习率衰减过快或梯度尺度不同导致收敛缓慢。RMSProp 通过对每个参数的历史梯度平方做指数加权平均，缓解了 Adagrad 学习率过快变小的问题。
- **方法与数学**：  
  $$
  E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma)g_t^2 \\
  \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t
  $$
  其中 $\gamma$ 通常取 0.9，$\epsilon$ 防止除零。
   - **代码示例（PyTorch）**：

    ```python
    optimizer = torch.optim.RMSprop(model.parameters  (), lr=0.01, alpha=0.9)

    ```
#### 1.2.6 Adam
- **原因**：Adam 结合了 Momentum 和 RMSProp 的思想，既考虑梯度的一阶矩（均值），也考虑二阶矩（方差），对每个参数自适应调整学习率，提升收敛速度和稳定性。
- **方法与数学**：  
  $$
  m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \\
  v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \\
  \hat{m}_t = \frac{m_t}{1-\beta_1^t} \\
  \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
  \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
  $$
  其中 $\beta_1=0.9$，$\beta_2=0.999$，$\epsilon=10^{-8}$。
- **代码示例（PyTorch）**：

  ```python
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  ```

#### 1.2.7 Warmup
- **原因**：在训练初期直接使用较大学习率可能导致模型不稳定，尤其是在大模型或大 batch 训练时。Warmup 通过逐步增加学习率，帮助模型稳定收敛。
- **方法**：前若干步（如前 5 个 epoch）将学习率从较小值线性或指数增加到目标学习率。
- **代码示例（PyTorch，使用 LambdaLR）**：

  ```python
    def warmup_lr_lambda(epoch):
      if epoch < 5:
          return float(epoch + 1) / 5
      return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)

  ```
---
## 2. Training loss 很小

模型至少已经把训练集学会了。接下来要看：

- **Testing loss 也小**<font color=green>(目标状态)  </font>
  - 完美情况，模型泛化能力强，无需大改。

- **Testing loss 很大**  
  - 这是最常见、也最容易困惑的情况，分为两类：

### 2.1 Overfitting（过拟合）

- **含义：** 训练集表现好，但模型“记住”了训练数据，没学到泛化规律。
  
#### 2.1.1 增强训练数据

- 可以通过收集更多的数据来扩充训练集，提升模型泛化能力。
- 也可以采用数据增强（data augmentation）技术，如缩放、对称（翻转），但通常不建议随意旋转，避免引入不合理的样本。
- 数据多样性提升后，模型更难“死记硬背”训练集，降低过拟合风险。

#### 2.1.2 降低模型复杂度

- 简化模型结构（如减少层数、参数数量），让模型表达能力受限。
- 这样模型只能学习到数据中的主要规律，难以记住训练集的噪声和偶然性，从而降低过拟合。
- 原因：复杂模型容易拟合训练集中的偶然特征，简单模型则更关注全局趋势。

#### 2.1.3 正则化（Regularization）

- 在损失函数中加入对参数的惩罚项，限制参数大小，常见有 L1 和 L2 正则化。
  - **L1 正则化**：损失函数加上参数绝对值之和（$ \lambda \sum |w_i| $），促使部分参数变为零，实现特征选择和稀疏性。
  - **L2 正则化**：损失函数加上参数平方和（$ \lambda \sum w_i^2 $），促使参数整体变小但不为零。
- **代码实现示例（以 PyTorch 为例）**：

  ```python
    # L2 正则化（weight decay）
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # L1 正则化（需手动加到 loss 上）
    l1_lambda = 1e-4
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    loss = original_loss + l1_lambda * l1_norm

  ```
- **为什么参数更小会让函数更平滑？**  
  - 以线性模型 $f(x) = w_1 x_1 + w_2 x_2 + \dots + b$ 为例，$w_i$ 越小，输入变化对输出的影响越弱，输出曲线变化更缓慢。对于深度网络，较小的权重意味着每层的输出不会因输入微小扰动而剧烈变化，整体函数对输入的响应更平滑。
  - 从数学上看，函数的导数（即对输入的敏感度）与参数成正比，参数越小，导数越小，曲线越平滑。
- **为什么我们相信更平滑的函数更可能正确？**  
  - 平滑的函数对未见样本更稳健，不容易因训练集中的偶然噪声而大幅改变预测结果。根据奥卡姆剃刀原则，简单、平滑的模型更有可能捕捉到数据的真实规律，而不是记住噪声。

#### 2.1.4 验证集的使用（Validation Set）

- 验证集是从训练数据中划分出来的一部分，不参与模型训练，仅用于评估模型在未见数据上的表现。
- 通过在验证集上监控 loss 和准确率，可以及时发现过拟合（训练集表现好但验证集变差）。
- 这样可以在训练过程中选择泛化能力最强的模型，防止模型只记住训练集。
- **验证集是在什么时候记录的？**  
  - 在每个 epoch（或每隔若干 epoch）训练结束后，用当前模型在验证集上计算 loss 和准确率，并记录下来。
  - 这样可以动态监控模型在未见数据上的表现，常用<font color=royalblue>“早停法”（early stopping）</font>在验证集性能最优时保存模型，防止过拟合。
- **解释 “This explains why machine usually beats human on benchmark corpora.”**  
  - 机器可以反复试错、微调参数，专门针对验证集优化，最终在标准数据集上表现优于人类。

#### 2.1.5 N-fold 交叉验证（N-fold Cross Validation）

- 将训练数据分成 N 份，每次用 N-1 份训练，剩下一份验证，循环 N 次，最终取平均效果。
- 这样可以充分利用数据，减少因数据划分带来的偶然性。
> PS: **为什么即使采用验证集依然不能完全避免过拟合？**  
  >- 如果验证集用得太多，模型选择过程本身也会“拟合”验证集（即在众多模型中挑选在验证集上表现最好的），这<font color=green>相当于对验证集进行了一种“梯度下降”</font>。
  >- 特别是模型复杂度极高（如 1000 层神经网络，每层 10 个特征）, 即使只用验证集选模型, 也可能过拟合验证集, 导致泛化能力下降。    

### 2️.2 Mismatch（数据分布不匹配）

- **含义：** 训练数据和测试数据本身不是同一个分布（如棚拍 vs 街拍、新闻 vs 微博）。
- **解决方案：** 关注数据本身，提升数据多样性和代表性。

---

## 3. 总结

- **Overfitting** → 右侧分支，关注模型复杂度与数据量
- **Small Gradient/Optimization** → 左侧分支，关注优化算法与参数设置

### 3.1 具体区分情况

| 问题               | Overfitting      | Small Gradient |
| ------------------ | ---------------- | -------------- |
| 表现               | Train 好，Val 差 | Train 也不好   |
| 本质               | 约束不足         | 优化受阻       |
| 是否学到了复杂函数 | 是，但不泛化     | 根本没学到     |
| 解决方向           | 限制模型         | 改善梯度流动   |


这张训练诊断决策树为模型训练过程中的问题定位和解决提供了“总地图”，帮助我们科学诊断和优化模型。

> 本文参考李宏毅老师机器学习课程，欢迎交流讨论。