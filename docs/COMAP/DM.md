---
layout: default
title: Decision Making
parent: COMAP
nav_order: 11
permalink: /COMAP/DM/
---

# 决策论，Decision Making

这里叙述决策论中两个重要的用于量化**决策标准**的**权重**的算法。

## 1.层次分析法，AHP

**层次分析法**（*Analytic Hierarchy Process*）通过引入专家的经验，估计不同决策标准之间的相对重要程度，最后计算各自的权重。由此可以得到，$$\mathtt{AHP}$$算法具有比较强的主观性，最后的效果与专家打分的结果直接相关。

假设我们现在对$$n$$个样本$$\{p_i\}$$分别有$$m$$个决策标准$$\{c_j\}$$，$$\mathtt{AHP}$$算法主要分为三步：

1. 通过两两比较，得到不同决策标准之间的相对重要程度矩阵$$A=(a_{ij})_{m\times m}$$，其中：
   
   $$
   a_{ij}=
   \begin{cases}
   &1,\quad &if\space c_i\space and\space c_j\space are\space of\space \mathbf{equal}\space importance\\
   &3,\quad &if\space c_i\space is\space \mathbf{weakly}\space more\space important\space than\space c_j\\
   &5,\quad &if\space c_i\space is\space \mathbf{strongly}\space more\space important\space than\space c_j\\
   &7,\quad &if\space c_i\space is\space \mathbf{very\space strongly}\space more\space important\space than\space c_j\\
   &9,\quad &if\space c_i\space is\space \mathbf{absolutely}\space more\space important\space than\space c_j\\
   &2,4,6,8\quad &intermediate\space values
   \end{cases}
   $$
   
   这里假定$$c_i$$的重要程度不低于$$c_j$$，另外有$$a_{ji}=\frac{1}{a_{ij}}$$，$$i\neq j$$，及$$a_{ii}=1$$。

   对矩阵$$A$$做一些标准化的操作，得到$$A^{'}=(a_{ij}^{'})_{m\times m}$$，其中：
   
   $$
   a_{ij}^{'}=\frac{a_{ij}}{\sum_{k=1}^ma_{kj}}
   $$
   
   最后得到各个决策标准的权重向量$$\mathbf{w}=\{w_1\space w_2\space \cdots\space w_m\}^T$$：
   
   $$
   w_i=\frac{\sum_{k=1}^ma_{ik}^{'}}{m}
   $$

2. 在得出权重向量$$\mathbf{w}$$后，我们还需要对初始的矩阵$$A$$做一致性检测。这里的一致性，指的就是最理想的情况下，有$$\forall i,j,k$$，有$$a_{ij}\cdot a_{jk}=a_{jk}$$。此检测的目的，最重要的是为了保证不会存在类似于这样的情况：$$c_i$$比$$c_j$$重要，$$c_j$$比$$c_k$$重要，$$c_k$$比$$c_i$$重要。

   定义一致性指标（*Consistency Index*）$$\mathtt{CI}=\frac{\lambda_{max}-n}{n-1}$$，其中$$\lambda_{max}$$表示矩阵$$A$$的特征值的最大值。取随机指标（*Random Index*）$$\mathtt{RI}$$，计算**一致性比率**（*Consistency Ratio*）$$\mathtt{CR}=\frac{\mathtt{CI}}{\mathtt{RI}}$$。其中$$\mathtt{RI}$$的取值通过查表得到：

   | $$\mathbf{m}$$  | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
   | --------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
   | $$\mathtt{RI}$$ | 0.00 | 0.00 | 0.58 | 0.90 | 1.12 | 1.24 | 1.32 | 1.41 | 1.45 | 1.49 |

   如果计算出的$$\mathtt{CI}<0.10$$，那么说明一致性较好，进而说明$$\mathtt{AHP}$$算法在此应用中是有效的。

3. 设针对$$n$$个样本$$m$$个决策标准的打分矩阵为$$P=(p_{ij})_{n\times m}$$，则最后各个样本的得分向量$$\mathbf{q}=P\cdot\mathbf{w}$$，其中$$\mathbf{q}=\{q_1\space q_2\space\cdots\space q_n\}^T$$，$$q_i$$表示第$$i$$个物品经由$$\mathtt{AHP}$$算法得到的综合得分。

## 2.熵权法，EWM

熵权法（*Entropy Weight Method*）通过衡量各决策标准的价值分散度来计算权重。一般来说，某个决策标准的分散程度越高，所具有的信息就越多，相应地就应该被赋予更高的权重，反之同理。

因此，熵权法是一种客观的，不依赖于专家打分等信息的算法。但是，在某些情况下，熵权法的结果可能会不准确，例如当测量值中的零值较多时。此外，熵权法只考虑了数值区分度，而忽略了指标的排序区分度。

同样，假设我们有$$n$$个样本及$$m$$个决策标准$$\{c_i\}$$，对应的打分矩阵$$P=(p_{ij})_{n\times m}$$，$$\mathtt{EWP}$$算法主要分为三步：

1. 首先对$$P$$中的各列做归一化，得$$P^{'}=(p_{ij}^{'})_{n\times m}$$，其中：
   
   $$
   p_{ij}^{'}=\frac{p_{ij}-\min_i p_{ij}}{\max_i p_{ij}-\min_i p_{ij}}\quad or\quad p_{ij}^{'}=\frac{\max_ip_{ij}-p_{ij}}{\max_i p_{ij}-\min_i p_{ij}}
   $$
   
   上两式中，具体选择前者还是后者需要考虑$$c_i$$与最后的综合得分是正相关还是负相关的。再计算对各个决策标准进行标准化后的矩阵$$Q=(q_{ij})_{n\times m}$$，其中：
   
   $$
   q_{ij}=\frac{p_{ij}^{'}}{\sum_{i=1}^np_{ij}^{'}}
   $$

2. 计算各个决策标准的熵值$$E_i$$如下：
   
   $$
   E_{i}=-\frac{\sum_{i=1}^nq_{ij}\ln q_{ij}}{\ln n}
   $$
   
   当$$q_{ij}$$的值越分散时，$$E_{i}$$的值越小。

   此外，考虑函数$$f(x)=x\ln x$$，由$$f^{''}(x)=\frac{1}{x}>0$$得$$f(x)$$为**下凸函数**，则由琴生不等式（*Jensen's Inequality*）得：
   
   $$
   E_i=-\frac{\sum_{i=1}^nf(q_{ij})}{\ln n}\leq-\frac{n\cdot f(\frac{\sum_{i=1}^nq_{ij}}{n})}{\ln n}=-\frac{n\cdot f(\frac{1}{n})}{\ln n}=1
   $$
   
   取等号当且仅当$$q_{1j}=q_{2j}=\cdots=q_{nj}=\frac{1}{n}$$。从而我们有$$E_i\in[0,1]$$。

   最后，计算权重向量$$\mathbf{w}=\{w_1\space w_2\space\cdots\space w_m\}$$，其中$$w_i$$表示由$$\mathtt{EWM}$$算法计算出的、对应于$$c_i$$的权重：
   
   $$
   w_i=\frac{1-E_i}{\sum_{i=1}^m (1-E_i)}
   $$

3. 计算各个样本的得分向量$$\mathbf{q}=P\cdot\mathbf{w}$$，其中$$\mathbf{q}=\{q_1\space q_2\space\cdots\space q_n\}^T$$，$$q_i$$表示第$$i$$个物品经由$$\mathtt{EWM}$$算法得到的综合得分。