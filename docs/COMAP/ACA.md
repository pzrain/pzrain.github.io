---
layout: default
title: Ant Colony
parent: COMAP
nav_order: 3
permalink: /COMAP/ACA/
---

# 蚁群算法（Ant Colony Algorithm，ACA）

## 算法原理

蚁群可以在没有任何提示的情况下找到食物源到巢穴的最短路径，并且能在环境发生变化后，自适应地搜索新的最佳路径。原因在于，蚂蚁会在走过的路径上释放一种信息素，路径上的信息素强度越大，蚂蚁选择该条路径的概率也就越高。由于最短路径具有的某种优势，会使得信息素在最短路径上的强度累积，这是一种**正反馈**的过程。对于单个蚂蚁，虽然它并没有主观上要寻找最短路径，但对于整个蚁群，又确实达到了寻找最短路径的客观效果。

## 算法步骤

1. 初始化相关参数

   * 蚁群规模

   * 转移概率
     $$ p_{ij}^k(j)=\left\{
     \begin{align}
     &\frac{[\tau_{ij}(t)]^{\alpha}\cdot[\eta_{ij}(t)]^{\beta}}{\sum_{s\in \mathit{allow_k}}[\tau_{is}(t)]^{\alpha}\cdot[\eta_{is}(t)]^{\beta}},\space&j\in\mathit{allow_k}\\
     &0,\space&j\notin\mathit{allow_k}
     \end{align}
     \right. $$
     $$\tau_{ij}(t)$$表示在$$t$$时刻城市$$i$$与城市$$j$$连接路径上的信息素浓度；$$\eta_{ij}(t)$$为启发函数，表示蚂蚁从城市$$i$$转移到城市$$j$$的期望程度。

     $$\alpha$$表示信息素的重要程度，即**信息素因子**；$$\beta$$表示启发函数的重要程度，即**启发函数因子**。

   * **信息素挥发因子**$$\rho$$
     $$
     \left\{
     \begin{align}
     &\tau_{ij}(t+1)=(1-\rho)\cdot\tau_{ij}(t)+\Delta\tau_{ij},\space0<\rho<1\\
     &\Delta\tau_{ij}=\sum_{k=1}^m\Delta_{ij}^k
     \end{align}
     \right.
     $$

     其中$$\Delta\tau_{ij}^k$$为第$$k$$只蚂蚁在城市$$i$$与城市$$j$$连接路径上释放信息素而增加的信息素浓度。

     $$
     \Delta\tau_{ij}^k=\left\{
     \begin{align}
     &\frac{Q}{L_k},若蚂蚁从城市i访问城市j\\
     &0, 否则
     \end{align}
     \right.
     $$
     
     其中，$$Q$$为**信息素常数**，表示蚂蚁循环一次所释放的信息素总量，$$L_k$$表示第$$k$$只蚂蚁经过路径的总长度。

   * 最大迭代次数$$iter\in[100,500]$$

   > $$\alpha\in[1,4],\beta\in\{3,4,5\},\rho\in[0.2,0.5],Q\in[10,1000]$$

2. 随机将蚂蚁放于不同的出发点，对每个蚂蚁计算其下一个访问城市，**直至所有蚂蚁访问完所有城市**。

3. 计算各个蚂蚁经过的路径长度$$L_k$$，同时对各个城市连接路径上的信息素浓度进行更新。

4. 迭代直至最大次数，或是最优解已经满足要求。

## 实例

依然以TSP问题为例子。

```matlab
clear;
rng default

D = distances;
n = cities;
m = 1.5 * n;  % number of ants;
alpha = 1;
beta = 5;
rho = 0.2;
Q = 10;
Heu_F = 1./D;
Tau = ones(n, n);
Table = zeros(m, n);
iter = 1;
iter_max = 100;
Route_best = zeros(iter_max, n);
Length_best = Inf(iter_max, 1);
Length_ave = zeros(iter_max, 1);
Limit_iter = 0;


while iter <= iter_max
    start = zeros(m, 1);
    for i = 1:m
        temp = randperm(n);
        start(i) = temp(1);
    end
    Table(:,1) = start;
    cities_index = 1:n;
    for i = 1:m
        for j = 2:n
            visited = Table(i,1:(j-1)); % cities that are already visited
            allow = ~ismember(cities_index, visited);
            allow = cities_index(allow);
            P = zeros(1, length(allow));
            for k=1:length(allow)
                P(k) = Tau(visited(end), allow(k))^alpha * Heu_F(visited(end), allow(k))^beta;
            end
            P = P / sum(P);
            Pc = cumsum(P); % cumulative summation
            target_index = find(Pc >= rand);   % Russian Roulette
            target = allow(target_index(1));
            Table(i,j) = target;
        end
    end
    Length = zeros(m,1);
    for i =1:m
        Route = Table(i,:);
        Length(i) = D(Route(n), Route(1));
        for j = 1:(n-1)
            Length(i) = Length(i) + D(Route(j), Route(j + 1));
        end
    end
    [min_Length, min_index] = min(Length);
    Length_ave(iter) = mean(Length);
    if iter == 1
        Length_best(iter) = min_Length;
        Route_best(iter,:) = Table(min_index,:);
        Limit_iter = 1;
    else
        Length_best(iter) = min(Length_best(iter-1), min_Length);
        if Length_best(iter) == min_Length
            Route_best(iter,:) = Table(min_index,:);
            Limit_iter = iter;
        else
            Route_best(iter,:) = Route_best((iter-1),:);
        end
    end
    
    Delta_Tau = zeros(n, n);
    for i = 1:m
        for j = 1:(n-1)
            Delta_Tau(Table(i,j), Table(i,j + 1)) = Delta_Tau(Table(i,j), Table(i,j + 1)) + Q / Length(i);
            Delta_Tau(Table(i,j+1), Table(i,j)) = Delta_Tau(Table(i,j+1), Table(i,j)) + Q / Length(i);
        end
        Delta_Tau(Table(i,n), Table(i,1)) = Delta_Tau(Table(i,n), Table(i,1)) + Q/Length(i);
        Delta_Tau(Table(i,1), Table(i,n)) = Delta_Tau(Table(i,1), Table(i,n)) + Q/Length(i); 
    end
    Tau = (1 - rho) * Tau + Delta_Tau;
    iter = iter + 1;
    Table = zeros(m, n);
end
```

## 适用范围

同样用于搜索一个比较好的局部最优解，各参数的取值对其影响较大，稳定性比较差。

对TSP及相应问题具有良好的适应性。