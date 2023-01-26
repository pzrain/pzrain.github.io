---
layout: default
title: Particle Swarm
parent: COMAP
nav_order: 6
permalink: /COMAP/PSO/
---

# 粒子群优化（Particle Swarm Optimization，PSO）

## 算法原理

粒子群算法来源于对一个简化社会模型的模拟（鸟群）。粒子群中的每一个粒子代表解空间中的一个解，其还具有速度以及加速度。粒子会按照速度在解空间中探索新的解，将其与自己得到的最优解以及群体得到的最优解进行比较，并根据此来调整此后的速度与加速度。

标准版本中，粒子群通过引入惯性权重$$w$$来控制**开发**与**探索**的平衡。其能够在没有得知太多问题信息的情况下，通过有效的搜索来在庞大解空间中找到候选解，但同时其也无法保证找到的最佳解为真实的最优解。

## 算法步骤

* **记号与迭代方程**

  第$$i$$个微粒表示为$$X_i=(x_{i1},x_{i2},\cdots,x_{iD})$$，其经过的最好位置（*Personal Best*，即适应度最大的位置）记为$$P_i=(p_{i1},p_{i2},\cdots,p_{iD})$$。群体中所有微粒经过的最好位置记为$$P_g$$。微粒$$i$$的速度记为$$v_i=(v_{i1},v_{i2},\cdots,v_{iD})$$。对每一代，微粒位置和速度的更新规则如下：

  $$
  \begin{align}
  v_{id}&\leftarrow w\cdot v_{id}+c_1\cdot\mathtt{rand}()\cdot(p_{id}-x_{id})+c_2\cdot\mathtt{rand}()\cdot(p_{gd}-x_{id})\\
  x_{id}&\leftarrow x_{id}+v_{id}
  \end{align}
  $$

  $$d=1,2,\cdots,D$$。其中$$w$$为**惯性权重**，$$c_1$$与$$c_2$$为**加速常数**，$$\mathtt{rand}()\in[0,1]$$为随机数。
  此外，更新过程中需要保持$$|v_{id}|\leq v_{max,d}$$。$$v_{max}$$决定了在当前位置与最好位置之间的区域的分辨率，如果$$v_{max}$$太高，微粒可能会飞过最好解；而如果$$v_{max}$$太小，微粒就不能进行足够的探索。为了防止发散，$$w$$必须比1小。通常$$c_1$$与$$c_2$$的取值范围为$$[1,3]$$。

  对$$v_{id}$$的更新分为三个部分。

  1. $$w\cdot v_{id}$$为微粒先前行为的惯性，为“**探索部分**”。
  2. $$c_1\cdot\mathtt{rand}()\cdot(p_{id}-x_{id})$$为“**认知部分**”，表示微粒本身的思考。参考1905年美国心理学家爱德华·桑代克提出的[效果律](https://zh.wikipedia.org/wiki/%E6%95%88%E6%9E%9C%E5%BE%8B)，大致概念是如果一个特定的特质成为一个能让个体存活下去的优势，那么这个特质将会持续存在。
  3. $$c_2\cdot\mathtt{rand}()\cdot(p_{gd}-x_{id})$$为“**社会部分**”，表示微粒间的信息共享与相互合作。参考班杜拉提出的[替代强化](https://wiki.mbalib.com/wiki/%E6%9B%BF%E4%BB%A3%E5%BC%BA%E5%8C%96)，大致可以理解为对榜样的强化，则周围其他人也会间接受到影响。

* **步骤**

  1. 初始化粒子数$$m$$，$$v_{max}$$，最大迭代次数$$G_{max}$$，并随机初始化粒子群的位置和速度。
  2. 评价每个微粒的适应度。
  3. 更新每个微粒的$$P_i$$以及群体的$$P_g$$。
  4. 更新每个微粒的$$X_i$$与$$v_i$$。
  5. 达到结束条件（迭代次数达到最大值或是找到了足够好的适应值）。

## 实例

### 1.简单的约束优化

$$
\min f(x,y)=x\cdot\exp(-\sqrt{x^2+y^2})
$$

直接调用`matlab`中的`particleswarm`函数即可。

```matlab
fun = @(x)x(1)*exp(-norm(x)^2);
lb = [-10,-15];
ub = [15,20];
[sol, fval] = particleswarm(fun, 2, lb, ub);
```

### 2.神经网络的参数优化

假设我们准备构建一个三层的神经网络。现在有一个分类任务，需要将一群鸟分为两类，输入参数有三个维度，分别是该群鸟的某种特征。因此神经网络的第一层有三个神经元，第三层有两个神经元，假设隐藏层有四个神经元，从而该神经网络共有$$3\times4+4\times2=20$$个参数，也即解空间的维数是20。设置其他参数如下：

```matlab
d = 20; % dimension of solution
m = 15; % number of particles
maxG = 80;
vmax = 0.5;
w = 0.5;
c1 = 1.5;
c2 = 1.5;
x = zeros(m,d);    % position
v = zeros(m,d);    % velocity
p = zeros(m,d);    % best position for particle
pg = zeros(1,d);   % best position global
vp = zeros(1,m);   % best value for particle
vg = -1;           % best value global
for i=1:m
    x(i,:) = rand(1,d) .* 3;
    v(i,:) = rand(1,d) * 0.5;
    p(i,:) = x(i,:);
    vp(i) = f(x(i,:));
    if (vp(i) > vg)
        vg = vp(i);
        pg = x(i,:);
    end
end
curG = 0;
```

PSO的更新迭代：

```matlab
% start PSO
while (curG < maxG)
    for i=1:m
    	% update velocity and position per particle
        v(i,:) = v(i,:) * 0.5 + ...
            c1 * rand(1,d).*(p(i,:) - x(i,:)) + ...
            c2 * rand(1,d).*(pg - x(i,:));
        v(i,v(i,:)>vmax) = vmax;   % clip velocity
        v(i,v(i,:)<-vmax) = -vmax;
        x(i,:) = x(i,:) + v(i,:);
        xv = f(x(i,:));
        if (xv > vp(i))
            vp(i) = xv;
            p(i,:) = x(i,:);
            if (vp(i) > vg)
                vg = vp(i);
                pg = p(i,:);
            end
        end
    end
    curG = curG + 1;
end
disp('optimal solution correct prediction = ')
disp(vg)
```

评价函数，也即预测准确度：

```matlab
function val = f(xi)
    train = [[1 2 3]; [2 4 3]; [4 5 2]; [0 1 4]; [7 4 3];
             [4 5 2]; [1 8 7]; [4 7 2]; [1 6 5]; [2 7 8]];
    label = [0 1 0 0 1 1 0 1 1 0];
    param_1 = xi(1:12);
    param_1 = reshape(param_1, 3, 4);
    param_2 = xi(13:20);
    param_2 = reshape(param_2, 4, 2);
    val = 0;
    for i=1:length(train)
        temp = train(i,:) * param_1;
        for j=1:length(temp)
            if (temp(j)) < 0
                temp(j) = 0;
            end
        end
        temp = temp * param_2;
        if ((temp(1) >= temp(2) && label(i) == 0) || ...
                (temp(1) < temp(2) && label(i) == 1))
            val = val + 1;
        end
    end
end
```

## 适用范围

粒子群算法也是隶属于进化算法。其要求解空间是**连续变化**的，因为粒子的速度和位置均是连续的。对于解空间离散的问题，例如TSP，粒子群优化算法可能不便直接应用。