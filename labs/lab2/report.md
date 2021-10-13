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
实验题目：逻辑回归
</font></center>


<br/>
<br/>

<center> <font size = 4> 学号：1190200523 </font></center>
<center> <font size = 4> 姓名：石翔宇 </font></center>

<div STYLE="page-break-after: always;"></div>

<!-- 此处用于换行 -->

## 一、实验目的

理解逻辑回归模型，掌握逻辑回归模型的参数估计算法。

## 二、实验要求及实验环境

### 实验要求

实现两种损失函数的参数估计（1. 无惩罚项；2. 加入对参数的惩罚），可以采用梯度下降、共轭梯度或者牛顿法等。

#### 验证方法

1. 可以手工生成两个分别类别数据（可以用高斯分布），验证你的算法。考察类条件分布不满足朴素贝叶斯假设，会得到什么样的结果。
2. 逻辑回归有广泛的用处，例如广告预测。可以到UCI网站上，找一实际数据加以测试。

### 实验环境

Windows 11 + Python 3.7.8

## 三、设计思想

### 1. 逻辑回归

逻辑回归又名对数几率回归，虽然名字叫“回归”，但实际上是一种分类学习方法。

考虑广义线性模型
$$
y=g(w^Tx+b)
$$
，除了可以做回归问题，还可以做分类任务，只需找到一个单调可微函数将分类任务的真是标记$y$与线性回归模型的预测值联系起来。

考虑连续可微的对数几率函数 
$$
y=\frac{1}{1+e^{-z}}
$$
将其代入式 1 得到
$$
y=\frac{1}{1+e^{-(w^Tx+b)}}
$$
将其取对数，则可以变化为
$$
\ln\frac{y}{1-y}=w^Tx+b
$$
若将$y$视为样本$x$作为正例的可能性，则$1-y$是其反例可能性，则称$\frac{y}{1-y}$为几率，$\ln\frac{y}{1-y}$为对数几率。实际上，就是在用线性回归模型的预测结果去逼近真实标记的对数几率。

现在我们来确定式 3 中的$w$和$b$。

我们将式 3 中的 $y$ 视为类后验概率估计 $p(y=1|x)$ ，则式 4 可重写为
$$
\ln\frac{p(y=1|x)}{p(y=0|x)}=w^Tx+b
$$
由式 3 可得
$$
p(y=1|x)=\frac{e^{w^Tx+b}}{1+e^{w^Tx+b}}\\
p(y=0|x)=\frac{1}{1+e^{w^Tx+b}}\\
$$
于是，我们可以用极大似然法估计$w$和$b$。给定数据集$\{(x_i,y_i\}^m_{i=1}$，为便于讨论，令$\beta=(w;b),\hat{x}=(x;1)$，则$w^Tx+b=\beta^T\hat{x}$。则对数几率回归模型最大化对数似然为
$$
l(w,b)=&\sum_{i=1}^{m}\ln p(y_i|x_i;w,b)\\
$$
要将$l(w,b)$最大化，也即最小化
$$
\begin{aligned}
J(\beta)=&-\frac{1}{m}\sum_{i=1}^{m}\ln ({y_i\frac{e^{\beta^T\hat{x}_i}}{1+e^{\beta^T\hat{x}_i}}+ ( 1-y_i)\frac{1}{1+e^{\beta^T\hat{x}_i}}})\\
	  =&-\frac{1}{m}\sum_{i=1}^{m}\ln \frac{y_ie^{\beta^T\hat{x}_i}+1-y_i}{1+e^{\beta^T\hat{x}_i}}\\
	  =&-\frac{1}{m}\sum_{i=1}^{m}({\ln (y_ie^{\beta^T\hat{x}_i}+1-y_i)}-\ln(1+e^{\beta^T\hat{x}_i}))\\
	  =&\frac{1}{m}\sum_{i=1}^{m}(-{y_i\beta^T\hat{x}_i+\ln(1+e^{\beta^T\hat{x}_i})})\\
\end{aligned}
$$
则我们的任务目标就变为
$$
\min_\beta J(\beta)
$$
若我们加入正则项，则有
$$
J'(\beta)=J(\beta)+\frac{\lambda}{2}\|\beta\|^2_2
$$
相应地，加入正则项后的任务目标变为
$$
\min_\beta J'(\beta)
$$

### 2. 梯度下降法（无正则项）

$J(\beta)$是关于$\beta$的高阶可导连续凸函数，则可用梯度下降法求导其最优解。

我们将$J(\beta)$对$\beta$求偏导可得
$$
\frac{\partial J(\beta)}{\partial\beta}=\frac{1}{m}\sum_{i=1}^m\hat{x}_i(-y_i+\frac{e^{\beta^T\hat{x}_i}}{1+e^{\beta^T\hat{x}_i}})=\nabla J(\beta)
$$
则按照梯度下降法，每一步我们都按照
$$
\beta_{i+1}=\beta_i-\alpha \nabla J(\beta)
$$
来更新 $\beta$ ，其中 $\alpha$ 被称作为学习率或者步长，满足 $\alpha > 0$ 。

### 3. 梯度下降法（有正则项）

相应地，加入正则项后，我们将$J'(\beta)$对$\beta$求偏导可得
$$
\frac{\partial J'(\beta)}{\partial\beta}=\frac{1}{m}\sum_{i=1}^m\hat{x}_i(-y_i+\frac{e^{\beta^T\hat{x}_i}}{1+e^{\beta^T\hat{x}_i}})+{\lambda}\beta=\nabla J'(\beta)
$$
每一步我们按照
$$
\beta_{i+1}=\beta_i-\alpha \nabla J'(\beta)
$$
来更新 $\beta$ 。

### 4. 牛顿法（无正则项）

$J(\beta)$是关于$\beta$的高阶可导连续凸函数，则也可用牛顿法求导其最优解。

假设 $\beta$ 的第$k$次迭代值为$\beta^{(k)}$，则可将$J(\beta)$在$\beta^{(k)}$附近进行二阶泰勒展开
$$
J(\beta)=J(\beta^{(k)})+g_k^T(\beta-\beta^{(k)})+\frac{1}{2}(\beta-\beta^{(k)})^TH(\beta^{(k)})(\beta-\beta^{(k)})
$$
其中$g_k=g(\beta^{(k)})=\nabla J(\beta^{(k)})$ 是$J(\beta)$的梯度向量在点$\beta^{(k)}$的值，$H(\beta^{(k)})$是$J(\beta)$的黑塞矩阵
$$
\begin{aligned}
H(\beta)=&\begin{bmatrix}\frac{\partial^2J}{\partial \beta_i \partial\beta_j}\end{bmatrix}_{n\times n}\\
		=&\frac{\partial^2J}{\partial \beta \partial\beta^T}\\
		=&\frac{1}{m}\sum_{i=1}^{m}\hat{x}_i\hat{x}_i^T\frac{e^{\beta^T\hat{x}_i}}{(1+e^{\beta^T\hat{x}_i})^2}\\
\end{aligned}
$$
在点$\beta^{(k)}$的值。函数$J(\beta)$有极值的必要条件是在极值点处梯度为$0$，即
$$
\nabla J(\beta)=0
$$
假设第$k+1$次迭代时有$\nabla J(\beta^{(k+1)})=0$，则由式 16 可得
$$
\nabla J(\beta)=g_k+H_k(\beta^{(k+1)}-\beta^{(k)})=0
$$
则迭代式为
$$
\beta^{(k+1)}=\beta^{(k)}-H^{-1}_kg_k
$$

### 4. 牛顿法（有正则项）

相应地，加入正则项后，$J'(\beta)$的黑塞矩阵为
$$
\begin{aligned}
H'(\beta)=&\begin{bmatrix}\frac{\partial^2J'}{\partial \beta_i \partial\beta_j}\end{bmatrix}_{n\times n}\\
		=&\frac{\partial^2J'}{\partial \beta \partial\beta^T}\\
		=&\frac{1}{m}\sum_{i=1}^{m}\hat{x}_i\hat{x}_i^T\frac{e^{\beta^T\hat{x}_i}}{(1+e^{\beta^T\hat{x}_i})^2}+\lambda\\
\end{aligned}
$$
则迭代式为
$$
\beta^{(k+1)}=\beta^{(k)}-H'^{-1}_kg'_k
$$


## 四、实验结果分析



## 五、结论



## 六、参考文献



## 七、附录:源代码(带注释)

