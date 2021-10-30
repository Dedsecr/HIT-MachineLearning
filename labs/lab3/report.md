<br/>
<br/>

<center> <font size = 5> 哈尔滨工业大学计算机科学与技术学院 </font></center>

<br/>
<br/>

<center> <font size = 7> 实验报告 </font></center>

<br/>
<br/>
<br/>

<center> <font size = 5> 
课程名称：机器学习 <br/>
课程类型：必修  <br/>
实验题目：实现k-means聚类方法和混合高斯模型
</font></center>



<br/>
<br/>

<center> <font size = 4> 学号：1190200523 </font></center>
<center> <font size = 4> 姓名：石翔宇 </font></center>

<div STYLE="page-break-after: always;"></div>

<!-- 此处用于换行 -->

## 一、实验目的

实现一个k-means算法和混合高斯模型，并且用EM算法估计模型中的参数。

## 二、实验要求及实验环境

### 实验要求

#### 测试

用高斯分布产生k个高斯分布的数据（不同均值和方差），其中参数自己设定。

（1）用k-means聚类，测试效果；

（2）用混合高斯模型和你实现的EM算法估计参数，看看每次迭代后似然值变化情况，考察EM算法是否可以获得正确的结果（与你设定的结果比较）。

#### 应用

可以UCI上找一个简单问题数据，用你实现的GMM进行聚类。

### 实验环境

Windows 11 + Python 3.7.8

## 三、设计思想

### 1. k-means算法

给定$n$个样本的集合$X=\{x_1, x_2,\dots,x_n\}$，$x_i\in \R^m$，k-means聚类的目标是将$n$个样本分到$k$个不同的类或簇中（假设$k<n$）。$k$个类$G_1,G_2, \dots,G_k$形成对集合$X$的划分，其中$G_i\cap G_j=\emptyset$，$\bigcup\limits_{i=1}^kG_i=X$。用$C$表示划分，一个划分对应着一个聚类结果。

k-means算法通过损失函数的最小化选取最优的划分$C^*$。

首先，我们将样本之间的距离$d(x_i, x_j)$定义为欧氏距离平方
$$
\begin{aligned}
d(x_i,x_j)=&\sum_{k=1}^m(x_{ik}-x_{jk})^2 \\
=&\|x_i-x_j\|^2
\end{aligned}
$$
我们定义损失函数$W(C)$为
$$
W(C)=\sum_{l=1}^k\sum_{C(i)=l}\|x_i-\bar{x}_l\|^2
$$
其中，$\bar{x}_l=\frac{1}{n_l}\sum\limits_{C(i)=l}x_i$ ，$n_l=\sum\limits_{i=1}^nI(C(i)=l)$。

则k-means算法就是求解最优化问题
$$
\begin{aligned}
C^*=&arg\min_CW(C) \\
=&arg\min_C\sum_{l=1}^k\sum_{C(i)=l}\|x_i-\bar{x}_l\|^2
\end{aligned}
$$
k-means算法是一个迭代的过程。首先，对于给定的中心值$(m_1,m_2,\dots,m_k)$，将每个样本指派到与其最近的中心$m_l$的类$G_l$中，得到聚类结果，使得目标函数极小化
$$
\min_{m_1,\dots,m_k}=\sum_{l=1}^k\sum_{C(i)=l}\|x_i-m_l\|^2
$$
然后，对于每个包含$n_l$个样本的类$G_l$，更新其均值$m_l$
$$
m_l=\frac{1}{n_l}\sum_{C(i)=l}x_i
$$
其中，$l=1, 2, \dots, k$。

重复上述两个步骤，直到$W(C)$结果小于阈值。




## 四、实验结果分析



## 五、结论



## 六、参考文献





## 七、附录:源代码(带注释)

