---
layout: default
title: Simulated Annealing
parent: COMAP
nav_order: 2
permalink: /COMAP/SA/
---

# 模拟退火算法（Simulated Annealing，SA）

用来寻找全局最优值，解决梯度下降等随机算法容易被困在局部最优值的问题。

## 算法原理

固体的退火过程：将固体加热到足够高的温度，再缓慢冷却，理论上，如果降温的过程**足够慢**，固体会一直保持在热平衡的状态，这样冷却达到低温时，将达到这一低温下的**最低内能状态**。（关键在于物理系统总是趋向于能量最低的稳定状态）

应用至最优化问题时，一般可以把温度$$T$$当作控制参数，目标函数值$$f$$视为内能$$E$$，而固体在某温度$$T$$时的状态对应一个解$$x_i$$。然后，算法试图控制参数$$T$$缓慢降低，使目标函数值$$f$$（内能$$E$$）也逐渐降低，直至趋于全局最小值，也即退火中低温时的最低能量状态。

## 算法步骤

1. 初始参数：初值$$T_0$$，终值$$T_f$$，控制参数$$T$$的衰减函数，Markov链的长度$$L_k$$（在某一温度下的迭代次数）

   > 常用衰减函数$$T_{k+1}=\alpha T_k$$，$$L=100n$$，$$n$$表示问题的规模。

2. 令$$T=T_0$$，随机生成一个初始解$$x_0$$，计算其内能$$E(x_0)$$

3. 更新$$T=T_i$$，对当前解$$x_i$$进行一个**扰动**，产生一个新解$$y$$，计算$$\Delta E=E(x_i)-E(y)$$。如果$$\Delta E<0$$，则新解$$y$$被接受，作为新的当前解；否则新解按照概率$$\exp(-\Delta E/T_i)$$被接受

   > 这里的3实际上是一个蒙特卡洛过程，理论上扰动的次数越多，最后的结果越好，但是耗时也更长。

4. 在每个温度$$T_i$$下，重复$$L_i$$次扰动，再更新$$T_i$$，直至达到$$T_f$$

> 需要注意的是，模拟退火在进行过程中有接受更差解的可能性，因此实际操作过程中通常会把过程中求出的最好解也一并记录下来。

最终能找到全局最优解，要求初始温度足够高，终止温度足够低，并且在任一温度的热平衡时间足够长。

随着温度的降低，算法会由原先的“大规模搜索”转变为“小范围搜索”。由接受概率也可以发现，当温度较低时，算法较难接受更差的解，因此要求在初始的广域搜索阶段，就要尽可能找到全局最优值所在的区域。

## 实例

### TSP问题

* 解空间$$S=\{(c_1,c_2,\cdots,c_n)\}$$
* 目标函数$$C(c_1,c_2,\cdots,c_n)=\sum_{i=1}^{n-1}d(c_i,c_{i+1})+d(c_1,c_n)$$
* 产生新解的扰动：
  1. 二变换：交换$$u,v$$有$$u<v<n$$
  2. 三变换：任选序号$$u,v,w$$，有$$u<v<w<n$$，将$$u$$和$$v$$之间的路径插入到$$w$$之后

以下是手写模拟退火算法求解TSP问题，设置初温为100，末状态温度为3，马尔科夫链长度为400，初始解为随机解`randperm(cities)`。

```matlab
rng default

% initialize parameters
a = 0.99;
t0 = 100;
tf = 3;
t = t0;
L = 400; % number of samples

load('usborder.mat', 'x', 'y', 'xx', 'yy');
hold on;
cities = 40;
locations = zeros(cities, 2);
n = 1;
while (n <= cities)
    xp = rand * 1.5;
    yp = rand;
    if inpolygon(xp, yp, xx, yy)
        locations(n,1) = xp;
        locations(n,2) = yp;
        n = n + 1;
    end
end

locations_x_tmp1 = locations(:,1) * ones(1, cities);
locations_x_tmp2 = locations_x_tmp1';
locations_y_tmp1 = locations(:,2) * ones(1, cities);
locations_y_tmp2 = locations_y_tmp1';
distances = sqrt((locations_x_tmp1 - locations_x_tmp2).^2 + ...
    (locations_y_tmp1 - locations_y_tmp2).^2);

sol_new = 1:cities;
E_current = inf; sol_current = randperm(cities);
E_best = inf; sol_best = 1:cities;
p = 1;

% start annealing
while t >= tf
    for r = 1:L
        sol_new = sol_current;
        if (rand < 0.5) % exchange two
            ind1 = 0; ind2 = 0;
            while (ind1 == ind2)
                ind1 = randCity(cities);
                ind2 = randCity(cities);
            end
            temp = sol_new(ind1);
            sol_new(ind1) = sol_new(ind2);
            sol_new(ind2) = temp;
        else % exchange three
            ind1 = randCity(cities - 3);
            ind2 = ind1 + randCity(cities - ind1 - 1);
            ind3 = ind2 + randCity(cities - ind2);
            templist = sol_new((ind1+1):(ind2-1));
            sol_new((ind1+1):(ind1+ind3-ind2+1)) = sol_new(ind2:ind3);
            sol_new(ind1+ind3-ind2+2:ind3) = templist;
        end
        E_new = distances(sol_new(cities), sol_new(1));
        for i = 1:(cities-1)
            E_new = E_new + distances(sol_new(i), sol_new(i+1));
        end
        if E_new < E_current 
            E_current = E_new;
            sol_current = sol_new;
            if E_new < E_best
                E_best = E_new;
                sol_best = sol_new;
            end
        else
            if rand < exp(-(E_new - E_current) ./ t)
                E_current = E_new;
                sol_current = sol_new;
            end
        end
    end
    t = t .* a;
end
disp('final distance')
disp(E_current)
disp('best solution:')
disp(sol_best)
disp('minimum distance:')
disp(E_best)

function res = randCity(cities)
    res = ceil(rand .* cities);
end
```

### Bound Constrained Minimization

调用`simulannealbnd`求解一类简单的定界优化问题

目标函数（`cam function`）
$$ \min f(x)=(4-2.1\cdot x_1^2+x_1^4/3)\cdot x_1^2+x_1\cdot x_2 + (-4+4\cdot x_2^2)\cdot x_2^2 $$

```matlab
rng default

ObjectiveFunction = @simple_objective;
x0 = [0.5 0.5];
lb = [-64 64];
ub = [64 64];
[x, fval, exitFlag, output] = simulannealbnd(ObjectiveFunction, x0, lb, ub);
fprintf('The best function value found is %g\n', fval);

function y = simple_objective(x)
    x1 = x(1);
    x2 = x(2);
    y = (4 - 2.1 * x1.^2 + x1.^4 ./3) .* x1.^2 + x1 .* x2 + (-4 + 4 * x2.^2) .* x2 .^ 2;
end
```

## 适用范围

大致与遗传算法类似，也需要能够对解空间进行方便编码以进行搜索。