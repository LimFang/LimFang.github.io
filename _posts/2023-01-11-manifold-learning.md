---
title: manifold_learning
date: 2023-01-11 11:30:00 +0800
categories: [machine learning, manifold]
tags: [math]     # TAG names should always be lowercase
math: true
mermaid: true
---

# manifold learning : what how and why
流形学习本质上就是非线性降维，但是不仅仅是减少数据量，更要反映出高维数据的几何特征，允许人们去**可视化、去噪和解释**这些数据

## 1.数学基础

### 符号

| Symbol                             | description                                                   |
|------------------------------------|---------------------------------------------------------------|
| $R_D$                              | D-dimensional Euclidean space                                 |
| $\mathcal{M}$                      | manifold                                                      |
| $C^{\mathcal{l}}$                  | functions with continuous derivatives up to order \mathcal{l} |
| $\mathcal{D}={\mathbf{\{x_i\}}_{i=1}^n}$ | A dataset containing n data points                      |

### 流形与嵌入

#### 流形与坐标卡

一个d维的流形 $\mathcal{M}$是满以下性质的拓扑空间：
- 对 $\mathcal{M} $上每一点p，存在映射 $\varphi $和p点的开集邻域 $U \subset \mathcal{M} $，满足 $\varPi: U\rightarrow\varPi(U) $是双射映射并且 $\varPi \varPi^{-1} $是光滑的。组合 $(U,\varphi) $称为卡，其逆映射称为局部坐标(local coordinate， $\mathcal{R}^d $映射到流形)。（**即任一点，它附近的领域，映射为欧氏空间，且双射**）
- 对于 $\mathcal{M} $上两点  $p,p^{'}\in\mathcal{M} $,并且两点分别包含在坐标卡 $(U,\varphi) (V,\phi) $中，如果 $U\cap V\ne \varnothing $，那么映射 $\varphi \circ \phi^{-1} $在 $\phi(U\cap V) $上是光滑的且存在光滑的逆映射

这里涉及到所谓的single、global 坐标卡的概念

#### 嵌入

微分几何中，嵌入是两个流形之间的光滑映射 $F:\mathcal{M} \rightarrow\mathcal{N} $，要求 $F^{-1} $存在且光滑。通常，高维数据存在于 $\mathcal{N}=\mathbb{R}^D $，需要估计的 $\mathcal{M} $是 $\mathbb{R}^D $的子流形。流形学习算法的关键可以视为寻找一个嵌入 $F:\mathcal{M} \rightarrow\mathbb{R}^m,d\leq m\ll D $,d是流形的本征维度

### Symmetric Positive Definite(SPD) 流形
SPD流形 $\mathcal{S}_{++}^{d} $是由所有的 $d\times d $的SPD矩阵组成的，即该流形上每个点都是矩阵

$$
\begin{gathered}\mathcal{S}_{++}^d=\{\mathbf{M}\in\mathbb{R}^{d\times d}:\mathbf{M}=\mathbf{M}^\top,x^\top\mathbf{M}\mathbf{x}>0,\forall\mathbf{x}\in\mathbb{R}^d\backslash\{\mathbf{0}_d\}\}\end{gathered}
$$

### Orthogonal matrices 正交矩阵流形 （i.e. special orthogonal (SO) group）

$$
\mathrm{SO}(n)=\{X \in \mathbb{R}^{n\times n}\mid XX^{\mathrm{T}}=X^{\mathrm{T}}X=\mathrm{I}_n\mathrm{~and~}\det(X)=1\}.
$$

### 黎曼流形和等距嵌入

#### 切空间和黎曼度量

- 在点 $p \in \mathcal{M} $的切空间 $\mathcal{T}_p \mathcal{M} $是 $\mathcal{M} $的切向量组成的d维向量空间，如果 $\gamma(t) $是 $\mathcal{M} $上通过p的光滑曲线，且 $\gamma(0)=p $，则这条曲线在p处的导数 $\gamma^{`}(0)$ 是  $\mathcal{T}_p \mathcal{M} $上的切向量。
- 黎曼度量将切线空间的内积与 $\mathcal{M} $上的每一点p相关联，具有黎曼度量的光滑流形定义为黎曼流形。对于切线空间的所有向量，其范数定义为 $\parallel \mathbf{x} \parallel_g = \sqrt{<\mathbf{v},\mathbf{v}>_g} $,距离定义为 $\parallel \mathbf{v}_1-\mathbf{v}_2 \parallel_g $，角度定义为 ${cos}^{-1}(<\mathbf{v}_1, \mathbf{v}_2>_g /(\parallel \mathbf{v}_1 \parallel_g\parallel \mathbf{v}_2 \parallel_g)) $

#### 等距和等距映射

- 已知光滑映射  $F:\mathcal{M} \rightarrow \mathcal{N} $，F在p的微分为 $dF_p: \mathcal{T} \mathcal{M} \rightarrow \mathcal{T} \mathcal{M}$，当我们固定坐标系统， $dF(p)$ 变成N*M的矩阵,将 $v \in {\mathcal{T}}\_{p} \mathcal{M}$ 映射到 ${dF}\_{pv} \in \mathcal{T}_{F(p)} \mathcal{M}$

- 当每个点p处的黎曼度量g由F保持时,即

![Desktop View](https://github.com/LimFang/LimFang.github.io/blob/main/assets/common/isometry.jpg?raw=true)

称两个黎曼流形 $(\mathcal{M},g,),(\mathcal{N},h) $间的映射是等距的





### 指数和对数映射

设唯一测地线 $\Gamma(t) $满足 $\Gamma(0)=p $，初始切向量 $\Gamma^{'}(0)=\mathbf{v} $，其中 $p \in \mathcal{M,\mathbf{v} \in \mathcal{T}_p \mathcal{M}} $，则在p处的指数映射定义为：

$$
Exp_p(v)=\Gamma(1)
$$

指数映射和对数映射是流形和它的切空间之间的同构映射，计算这些映射的算法依赖于感兴趣的流形和切空间的基点。
给定SPD矩阵 $\mathcal{M} \in {\mathcal{S}_{++}^{d}}$，

对应的**矩阵对数函数**为 $\mathrm{logm}(\mathbf{M}):\mathcal{S}_{++}^d \rightarrow sym(d)$:

$$
{logm}(M)=U\log(\Sigma)U^\top 
$$

sym(d)代表dxd的对称矩阵张成的空间， $\ddot{U\Sigma U}^{\top}=M $
- 给定对称矩阵 $N \in sym(d) $，**矩阵指数函数**为 $expm(\mathbf{N}:sym(d)\rightarrow \mathcal{S}_{++}^d) $：

$$
{expm}(\mathbf{N})=\mathbf{U}\exp(\mathbf{\Sigma})\mathbf{U}^\top 
$$

 其中 $\ddot{U\Sigma U}^{\top}=\mathbf{N} $
### weighted Frechet mean (wFM)

给定黎曼流形  $(\mathcal{M},g)$ ，N个数据点 ${X_i}_{i=1}^N \subset\mathcal{M}$，

具有凸约束的权重 ${w_i}_{i=1}^N \subset(0,1]$,WFM可以定义为:找到流形上的一个点，使得加权方差的最小化：

$$wFM({X_i},{w_i})=argmin_{m \in\mathcal{M}}\sum_{i=1}^N w_id^2(X_i,m)$$
平均权重时，FM常在流形卷积、激活和归一化层中使用，作为经典欧氏均值的推广。


### 收回操作 （Retraction operation）
- SPD上的收回操作,在局部刚性的条件下，从切空间回到流形上的光滑映射 $\Gamma_{M} (\cdot):{\mathcal{T}}\_{M} \mathcal{S}_{++}^d \to {\mathcal{S}}\_{++}$

$$
\Gamma_M(\zeta)=M^{\frac12} \text{expm} (M^{-\frac12} \zeta M^{-\frac12})M^{\frac12} 
$$

  其中， $\zeta \in \mathbb{R}^{d \times d} $是切空间M上的点
  
### 正交投影 （Orthogonal projection）
黎曼流形上的正交映射是将点M的任意一个梯度转换为切空间上的黎曼梯度 ${\pi}\_{\mathbf{M}}(\cdot):\mathbb{R}^{d\times d}\rightarrow {\mathcal{T}}\_{M} {\mathcal{S}}_{++}^{d} $

$$
\pi_M(\nabla_M)=M\frac12(\nabla_M+\nabla_M^\top)M
$$

## 2.流形学习的前提与范式
### 流形假设
假设数据是从嵌入在 $\mathbb{R}^D $中的d维流形M上或其附近的分布P采样的。这就是流形假设。

### 范式1 Neighborhood graphs
反映出数据的局部几何和拓扑信息；用两种方式定义邻居：
- radius-neighbor graph
$$如果 \parallel x_i -x_j\parallel \leq r，那么x_j是 x_i的邻居$$
- KNN 图
  如果 $x_j$ 是最靠近 $x_i$ 的前k个点 ，那么 $x_j$ 是 $x_i$ 的邻居
可以用以下的kernel function定义图的kernel matrix来衡量图节点间的权重，K通常是稀疏的

$$
\begin{align}
K_{ij} &= K(\frac{\parallel x_i -x_j \parallel}{h}), x_j \in \mathcal{N}_i  \\
 &=   0,    otherwise \\
\end{align}
$$ 

### 范式2 Linear local approximation
局部线性近似对于单变量函数来说，即一个微分函数可以被近似为其切线；一个多元函数，他的线性近似可以看作在该切点附近的切平面

![Desktop View](https://github.com/LimFang/LimFang.github.io/blob/main/assets/common/multiple_lla.jpg?raw=true)

该方法从PCA和random projection演化来，后两者只关注全局的线性信息，没有利用参考点x附近的几何结构，常利用加权PCA实现（IPCA）：

$$
C=\frac{1}{n} \sum_{i=1}^n(x_i-x)(x_i-x)^T
$$

### 范式3 Principal curves and principal d-manifolds

### 范式4 Embedding algorithms
嵌入算法的任务是生成输入的平滑映射，从而尽可能地减少邻域信息的失真
#### PCA

$$
\min_{\mathbf{T}:\mathbf{T}\in\mathbb{R}^{D\times d},\mathbf{T}^{\top}\mathbf{T}=\mathbf{I}_d}\sum_{i=1}^n\lVert x_i-\mathbf{T}\mathbf{T}^{\top}x_i\rVert^2=\min_{\mathbf{T}:\mathbf{T}\in\mathbb{R}^{D\times d},\mathbf{T}^{\top}\mathbf{T}=\mathbf{I}_d}\lVert\mathbf{X}-\mathbf{X}\mathbf{T}\mathbf{T}^{\top}\rVert_F^2 
$$

#### “One shot” embedding 
- Isomap
![Desktop View](https://github.com/LimFang/LimFang.github.io/blob/main/assets/common/isomap.jpg?raw=true)

- Diffusion Maps/Laplacian Eigenmaps
  谱聚类(spectral embedding)会将图拉普拉斯矩阵和Laplace-Beltrami算子联系起来
  
![Desktop View](https://github.com/LimFang/LimFang.github.io/blob/main/assets/common/dissision_map.jpg?raw=true)

-构造图拉普拉斯矩阵

![Desktop View](https://github.com/LimFang/LimFang.github.io/blob/main/assets/common/construct_graph_Laplacian_matrix.jpg?raw=true)

- Local Tangent Space Alignment (LTSA)
  
![Desktop View](https://github.com/LimFang/LimFang.github.io/blob/main/assets/common/Local_tangent_space_alignment.jpg?raw=true)

#### “Horseshoe” effects
基于特征向量的方法，在数据流形具有较大的纵横比时，此类算法会失败，被称为Repeated Eigendirection Problem，在真实数据集中普遍存在。吸引-排斥算法，如t-SNE可以克服此类问题

## 3.流形学习的统计学基础
### Biases in ML. Effects of sampling density and graph construction
### Choosing the scale of neighborhood
### Estimating the intrinsic dimension
### **Estimating the Laplace-Beltrami operator**
### Embedding distortions

## 4.针对流形数据的神经网络

### 流形下的复数表示
对于复数 $\forall z=x+iy\in\mathbb{C} $,x、y为实数，其极坐标为：
$$
\begin{aligned}
&z=r(x,y)\exp(i\theta(x,y)) \\
&\mathrm{abs}(\mathrm{z})=r(x,y)=\sqrt{x^2+y^2} \\
&\mathrm{pha(z)}=\theta(x,y)=\arctan(y,x)
\end{aligned}
$$
幅值函数视作1X1的SPD矩阵

相位函数看作1维圆环，符合SO(2)
### 流形下的特征融合 （欧氏下的特征融合对比）
如Chakraborty等人所证明的，卷积WFM层与复值标度是等变的，因为放缩旋转群 $\mathcal{R}^{+}\times SO(2) $是等距的，即它传递地作用在复平面C上。（等距群的等价性）

#### 流形融合：输出是以黎曼距离和切线空间上的切线向量表示的特征图
距离融合：输入为 ${X_i}\_{i=1}^{N} \subset \mathcal{M}$ ,输出为 ${d(X_i,M)}\_{i=1}^N \subset \mathbb{R}^N $，其中 $M=\mathrm{FM}(\{X_i\})$
切线融合：定义为在固定在恒等点的切线空间上对流形数据使用实值CNN的线性层，输入为 $\{X_i\}\_{i=1}^{N} \subset \mathcal{M}$，产生切向量 $\mathrm{Log}\_\mathrm{Id}(\{X\_\mathrm{i}\})T\_\mathrm{Id} \mathscr{M} $, 加上一个作为映射的实值神经网络 $\mathrm{NN}:T_{\mathrm{ld}} \mathcal{M} \to \mathbb{R}^N $，输出为 $f:\mathcal{M} \rightarrow {\mathbb{R}}^N$，即, $f={NN \circ Log}_{\mathbf{ld}}$ 将切向量转换为实数特征
 
#### 欧氏融合
在欧几里得特征级融合互补的实值输出特征图，黎曼距离+局部切向量在通道维度连接

### 流形下的激活函数
通常，通过将欧几里德激活函数集成到相应的切线空间中来形成流形激活函数有三个步骤：
1) 应用对数映射从流形M上的一点到其切线空间上的一点；
2) 在切线空间上执行欧几里得非线性函数；
3) 应用指数映射从切线空间返回到M

#### Tangent ReLU（tReLU）
 其实就是将流形上的点映射到切空间，再执行非参的非线性操作
 
$$
X\in\mathcal{M}\stackrel{tReLU}{\longrightarrow}\mathrm{Exp}_{\mathrm{Id}}(\mathrm{ReLU}(\mathrm{Log}\_{\mathrm{Id}} (X)))\in\mathcal{M}
$$ 
  
#### Tangent PeLU（tPReLU）

将流形上的点映射到切空间，再执行参数化的非线性操作 

$$
X\in \mathcal{M}\stackrel{tPReLU}{\longrightarrow}\mathrm{Exp}_{\mathrm{Id}}(\mathrm{PReLU}(\mathrm{Log}\_{\mathrm{Id}}(X)))\in\mathcal{M}
$$
 
#### G-trans

这是一种不符合上述三步过程的的参数化激活函数，它是专门为 $\mathcal{S}_{++}^1 \times SO(2) $流形网络设计的，每个特征通道学习一个缩放参数和一个旋转参数，

### 流形下的批归一化函数
基于矩阵李群（Lie Groups）上的高斯分布，Chakraborty在流形范数下提出了一种封闭形式的黎曼批量归一化算法,SPD矩阵流形上的**高斯分布**将变为对数正态分布

![Desktop View](https://github.com/LimFang/LimFang.github.io/blob/main/assets/common/BN_Lie_Groups.jpg?raw=true)

在神经网络应用中，将Spd和SO(2)看作李群，然后以流形范数（ManifoldNorm）中应用归一化算法，当流形数据不符合高斯分布时采用此种方法反而会导致性能的下降

## reference
Interpreting Posterior of Gaussian Process for Regression from https://medium.com/
