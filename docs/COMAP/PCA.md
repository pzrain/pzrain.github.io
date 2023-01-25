---
layout: default
title: Principle Component Analysis
parent: COMAP
nav_order: 4
permalink: /COMAP/PCA/
---

# 主成分分析（Principal Component Analysis，PCA）

## 基本原理

在多元统计分析中，主成分分析是一种统计分析、简化数据集的方法，其利用**正交变换**来对一系列可能相关的变量的观测值进行线性变换，从而投影为一系列线性不相关变量的值，这些不相关变量称为主成分。

* 将坐标轴中心移到数据的中心，然后旋转坐标轴，使得数据在$$C_1$$轴上的方差最大，即全部$$n$$个数据个体在该方向上的投影最为分散，也就意味着更多的信息被保留下来。$$C_1$$成为第一主成分。
* 找一个$$C_2$$，使得$$C_2$$与$$C_1$$的协方差为0，以免与$$C_1$$信息重叠，且使得数据在该方向上方差尽量大。$$C_2$$成为第二主成分。
* 以此类推，找到第三、第四个$$\cdots$$主成分。$$p$$个随机变量可以有$$p$$个主成分。

> 和线性代数里的主轴定理比较类似，实际上是找到特征向量

PCA对变量的缩放很敏感。

## 算法步骤

1. **对原始数据进行标准化处理**（原因在于，PCA对变量的缩放敏感）。
2. 计算样本的相关系数矩阵$$R$$。
3. 计算相关系数矩阵$$R$$的特征值$$(\lambda_1,\lambda_2,\cdots,\lambda_p)$$和相应特征向量$$\mathbf{a}_i=(a_{i1},a_{i2},\cdots,a_{ip})$$。
4. 计算贡献率$$\frac{\lambda_t}{\sum_{i=1}^p\lambda_i}$$，通过贡献率大小选择前$$k$$个主成分。
5. 计算主成分得分。

## 实例

```matlab
[coef, score, latent] = pca(A);
```

注意，这里$$A$$应当是标准化后的矩阵。

返回值中，`coef`表示各主成分的相对于原数据各列的系数，`score`为主成分得分（$$\mathit{score}_{ij}=A(i,:)\cdot coef(:,j)$$），`latent`为主成分方差，也即贡献率，默认情况按照从大到小排列。

在得到主成分得分后，还需根据`coef`考虑主成分各列的实际含义，做出分析。

```matlab
clc
clear
A = xlsread('Coporation_evaluation.xlsx', 'B2:I16');

a = size(A,1);
b = size(A,2);
disp([a b])
for i =1:b
    SA(:,i) = (A(:,i) - mean(A(:,i))) / std(A(:,i));
end

CM = corrcoef(SA);
[V, D] = eig(CM);
DS = zeros(b, 3);
for j = 1:b
    DS(j,1) = D(b+1-j, b+1-j);  % descending order
end
for i = 1:b
    DS(i,2) = DS(i,1) / sum(DS(:,1)); % contribution
    DS(i,3) = sum(DS(1:i,1)) / sum(DS(:,1)); % cumulative contribution
end

T = 0.9;
for k = 1:b
    if DS(k,3) >= T
        Com_num = k;
        break
    end
end

PV = zeros(b, Com_num);
for j = 1:Com_num
    PV(:,j) = V(:,b+1-j); % V here may be wrong...
end

new_score = SA*PV;
total_score = zeros(a,2);
for i = 1:a
    total_score(i,1) = sum(new_score(i,:));
    total_score(i,2) = i;
end
```



