---
title: Gaussian + Bayesian + ODE + rinemmeina manifold + Geometry OF Complex Numbers
date: 2023-01-15 11:30:00 +0800
categories: [machine learning, manifold]
tags: [deep learning]     # TAG names should always be lowercase
math: true
---

# 1.SCALING–ROTATION MANIFOLD FOR THE GEOMETRY OF COMPLEX NUMBERS

## Scenes and Background
如何处理流形数据,并将非欧几里德几何融入深度学习?

例如，在MR和合成孔径雷达图像中，一张图片的像素强度值可以通过复数 $s=me^{j\theta}$ 进行任意缩放，其中所有像素值同时按m进行幅度缩放并通过 $\theta$ 进行相位移位，实部和虚部之间的独立性假设此时不成立，采用两个实数CNN会导致这种相关性信息缺失

> any measurement z is simply a representative of a whole class of possible equivalent measurements {zs : s = mejθ : m > 0, ∀θ }. Instead of being independent of each other, the real and imaginary components (x, y) of z covary in this equivalent class.

- 平移群（group of translations）：标准卷积（欧氏空间）、旋转（超球面）
- 传递性（transitivity）：群G在空间S上的作用是传递的，如果空间S上存在从一点到另一点的g∈G；传递作用在复平面的非欧几里德空间上的群是复平面中的非零放缩和平面旋转
- 同变性（equivariance）：系统在不同位置的工作原理相同，但它的响应随着目标位置的变化而变化 
- 不变性（invariance）：

### 流形下的复数表示

对于复数 $\forall z=x+iy \in \mathbb{C}$ ,x、y为实数，其极坐标为：

$$
\begin{aligned}
&z=r(x,y)\exp(i\theta(x,y)) \\
&\mathrm{abs}(\mathrm{z})=r(x,y)=\sqrt{x^2+y^2} \\
&\mathrm{pha(z)}=\theta(x,y)=\arctan(y,x)
\end{aligned}
$$

### 放缩旋转 积流形
在上一点的基础上，将非零复平面 $\tilde{C}$ 视作**非零放缩**和**2D旋转**的积空间

$$
\tilde{C} \leftrightarrow \mathbf{R}^+ \times SO(2)
$$

- 幅值函数视作1X1的SPD矩阵 $\mathbf{R}^+$ 

- 相位函数看作1维圆环，符合SO(2)(旋转李群)
$$
\begin{aligned}\mathbf{z}&=|\mathbf{z}|\exp(i\not\leq\mathbf{z})\underset{F^{-1}}{\operatorname*{\overset{F}{\operatorname*{\longleftrightarrow}}}}(|\mathbf{z}|,R(\measuredangle\mathbf{z}))\\R(\measuredangle\mathbf{z})&=\begin{bmatrix}\cos(\theta)&-\sin(\theta)\\\sin(\theta)&\cos(\theta)\end{bmatrix}.\end{aligned}
$$

- 一般而言矩阵的指数和对数如下定义：
$$
\begin{aligned}
\operatorname{expm}(X)& =\sum_{n=0}^\infty\frac{X^n}{n!}  \\
\text{X}& =\mathrm{logm}(Y)\text{ if and only if }Y=\mathrm{expm}(X) 
\end{aligned}
$$

- 在积流形中距离定义为 
$$
d(\mathbf{z}_1,\mathbf{z}_2)=\sqrt{\log^2\frac{|\mathbf{z}_2|}{|\mathbf{z}_1|}+\|\operatorname{logm}(R(\measuredangle\mathbf{z}_2)R(\measuredangle\mathbf{z}_1)^{-1})\|^2}
$$

#### 旋转放缩在复平面上的传递性(transitive)
定义在流形上的传递性群action：黎曼流形 $\mathcal{M}$ 和群G，单位元为e，如果存在映射 $L:G\times \mathcal{M} \rightarrow \mathcal{M}$ ,即 $(g,X) \longmapsto g.X$ 满足以下两种条件：

$$
\begin{aligned}
&\textit{Identity: L}(e,X)=e.X=X. \\
&Compatibilitv: (gh).X=g.(h.X) \forall g,h\in G
\end{aligned}
$$

当且仅当给定 $X,Y \in \mathcal{M}$，存在一个元素 $g \in G$ 使得 $Y=g.X$ 时，称该action为transitive的。


#### 旋转放缩在复平面上的等距性(isometric)
即证明，在复平面 ${\mathbf{\tilde C}}$ 放缩和旋转能够保证流形距离, $\forall\mathbf{z}_1,\mathbf{z}_2\in\widetilde{\mathbf{C}},g\in\mathbf{R}^+\times\mathrm{SO}(2)$ :

$$
d(g.\mathbf{z}_1,g.\mathbf{z}_2)=d(\mathbf{z}_1,\mathbf{z}_2)
$$

## Challenges
保证分类器的尺度不变性，常见方法是使用复值放缩来增强训练数据，但是会导致低效和训练时间过长的问题。


## Solutions
用极坐标形式表示一个复数，并定义一个黎曼流形，在这个流形上，复值放缩对应于一般的传递作用群。当数据样本位于黎曼流形上时，可用先前已建立的深度学习结果。

### 卷积操作
- 欧氏空间上的标准卷积是实数的加权平均

$$
\{w_k\}*\{x_k\}=\sum_{k=1}^nw_kx_k
$$

- 给定黎曼流形 $(\mathcal{M},g)$ ，N个数据点 ${X_i}_{i=1}^N \subset\mathcal{M}$，具有凸约束的权重${w_i}_{i=1}^N \subset(0,1]$ ,WFM可以定义为:找到流形上的一个点，使得加权方差的最小化：

$$
wFM({X_i},{w_i})= argmin_{m \in\mathcal{M}}\sum_{i=1}^N w_id^2(X_i,m)
$$

定义复数wFM卷积 $\tilde{*}$ ，权重为实数通过随机梯度下降学习，输出为复数，是一个加权最小均方误差过程，与复数放缩等变



### 全连接层函数
在分类任务中，CNN的最终表示是不随每一类的变化而变化的。需要定义一个复平面 $\tilde{\mathbf{C}}$ 上的FC函数，针对空间 $\mathbf{R}^+ \times SO(2)$ 的action具有不变性;流形距离d天然具有这种不变性，提出将对G等变的集合中每个点到它们的wFM的距离，定义为复平面上的一个新的FC函数，Distance Transform FC Layer:

考虑s像素k通道输入，总共 $s \times c$ 个复数值，设计 $s \times c$ 个权重，首先计算出wFM点 $\mathbf{m}$ ，再计算每一个点到 $\mathbf{m}$ 的距离

$$
\begin{aligned}\mathbf{m}&=\{w_k\}\widetilde{*}\{\mathbf{t}_k\}\\u_k&=d(\mathbf{t}_k,\mathbf{m}).\end{aligned}
$$

### 流形下的非线性激活函数
通常，通过将欧几里德激活函数集成到相应的切线空间中来形成流形激活函数有三个步骤：
1) 应用对数映射(log for $r \in \mathbf{R}^+$ and logm for $R(\theta) \in SO(2)$)从流形M上的一点到其切线空间上的一点；
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

这是一种不符合上述三步过程的的参数化激活函数，它是专门为 $\mathcal{S}_{++}^1 \times SO(2) $流形网络设计的，每个特征通道学习一个缩放参数和一个旋转参数,对应于在特定深度层为每个通道学习一个复值乘数

### 流形下的批归一化函数
基于矩阵李群（Lie Groups）上的高斯分布，Chakraborty在流形范数下提出了一种封闭形式的黎曼批量归一化算法,SPD矩阵流形上的**高斯分布**将变为对数正态分布

![Desktop View](https://github.com/LimFang/LimFang.github.io/blob/main/assets/common/BN_Lie_Groups.jpg?raw=true)

在神经网络应用中，将Spd和SO(2)看作李群，然后以流形范数（ManifoldNorm）中应用归一化算法，当流形数据不符合高斯分布时采用此种方法反而会导致性能的下降

### 流形下的残差连接
出现矛盾：我们可以在欧氏空间中相加两个向量，但不能在非欧氏流形上相加两个点。

考虑两个特征层 $f_1(s_1,c_1),f_2(s_2,c_2),s_1<s_2$ ，二者存在一个skip connection，通过wFM卷积将 $f_2$ 的空间维度从 $s_2$ 变为 $s_1$ ,在在特征维度c上进行对其，得到 $f(s_1,c_1+c_2)$


### 流形下的特征融合 （欧氏下的特征融合对比）
如Chakraborty等人所证明的，卷积WFM层与复值标度是等变的，因为放缩旋转群 $\mathcal{R}^{+}\times SO(2) $是等距的，即它传递地作用在复平面C上。（等距群的等价性）

#### 流形融合：输出是以黎曼距离和切线空间上的切线向量表示的特征图
距离融合：输入为 ${X_i}\_{i=1}^{N} \subset \mathcal{M}$ ,输出为 ${d(X_i,M)}\_{i=1}^N \subset \mathbb{R}^N $，其中 $M=\mathrm{FM}(\{X_i\})$
切线融合：定义为在固定在恒等点的切线空间上对流形数据使用实值CNN的线性层，输入为 $\{X_i\}\_{i=1}^{N} \subset \mathcal{M}$，产生切向量 $\mathrm{Log}\_\mathrm{Id}(\{X\_\mathrm{i}\})T\_\mathrm{Id} \mathscr{M} $, 加上一个作为映射的实值神经网络 $\mathrm{NN}:T_{\mathrm{ld}} \mathcal{M} \to \mathbb{R}^N $，输出为 $f:\mathcal{M} \rightarrow {\mathbb{R}}^N$，即, $f={NN \circ Log}_{\mathbf{ld}}$ 将切向量转换为实数特征
 
#### 欧氏融合
在欧几里得特征级融合互补的实值输出特征图，黎曼距离+局部切向量在通道维度连接


## Reference
[1] R. Chakraborty, Y. Xing, and S. X. Yu, “SurReal: Complex-Valued Learning as Principled Transformations on a Scaling and Rotation Manifold,” IEEE Trans. Neural Netw. Learning Syst., vol. 33, no. 3, pp. 940–951, Mar. 2022, doi: 10.1109/TNNLS.2020.3030565.


# 2.Cholesky decomposition on Riemannian manifold for ODE solve

## Scenes and Background
- 针对时间序列的非欧几里得数据建模，为下游的分类、分割、预测等任务做准备；目前常使用卷积神经网络(CNN)或递归神经网络(RNN)来捕获时间序列数据的空间或（长、短）时间特征
- 对称正定矩阵能够用于学习有益的统计表示，在计算机视觉、信号处理和医学图像分析中得到了积极的研究
- 现实世界中的数据主要是不规则采样的数据，可能是由于记录或设备中的意外错误，流形学习的采用提高了模型对噪声和离群点的影响的鲁棒性

### Choleskt space 下的黎曼流形与度量[2]

下文中， $\lfloor \cdot \rfloor$ 和 $D(·)$ 表示矩阵的严格下三角和对角部分，

所有对角元素为正的下三角矩阵 $\mathcal{L}$ 张成空间 $\lfloor \mathcal{L} \rfloor ={\lfloor \mathcal{X} \rfloor} |  X \in \mathcal{L} \in \mathbb{R}^{d(d-1)/2}$ 的子流形 $\mathcal{L}_{+}$ 定义为Cholesky space; $L_{+}$ 在给定矩阵  $L \in L_{+}$ 处的切线空间与线性空间 $\mathcal{L}$ 相同，有Frobenius内积 $\langle X,Y\rangle_{F}=\sum_{i.i=1}^{d}X_{ij}Y_{ij},\forall X,Y\in\lfloor\mathcal{L}\rfloor$ 和 $\langle\mathcal{D}(L)^{-1}\mathcal{D}(X),\mathcal{D}(L)^{-1}\mathcal{D}(Y)\rangle_F$ ，其中 $\mathcal{D}({\mathcal{L}})=\{\mathcal{D}(X)|X\in\mathcal{L}\}$ 

子流形 $\mathcal{L}_{+}$ 切空间 $\mathcal{T}_L \mathcal{L}_{+}$ 上的度量 $\tilde{g}$为：

$$
\begin{gathered}
\tilde{g}_{L}(X,Y) \begin{aligned}=\langle X,Y\rangle_F+\langle\mathcal{D}(L)^{-1}\mathcal{D}(X),\mathcal{D}(L)^{-1}\mathcal{D}(Y)\rangle_F\end{aligned} \\
=\sum_{i=1}^d\sum_{j=1}^iX_{ij}Y_{ij}+\sum_{j=1}^dX_{jj}Y_{jj}L_{jj}^{-2}. 
\end{gathered}
$$

一个矩阵 $S \in \mathcal{S}_d^+$ ，是SPD流形上的一个点，按Cholesky 分解为下三角矩阵L和其转置的乘积 $S=L\times L^{T}$；Cholesky map 在SPD流形和子流形 $\mathcal{L}_{+}$ 之间是**微分同胚**

> A diffeomorphism is a continuous, invertible, structure- preserving map between two differentiable surfaces
>
> 两个流形是微分同胚的意味着这两个流形拥有拓扑同胚的拓扑结构，然后在局域坐标中两个流形局域是一样的差别仅仅在一个可微的坐标变换，就是说，这两个流形大体上和局域上看起来都没差别（差别仅在一个映射变换上）


### 指数与对数映射

设唯一测地线$\Gamma(t)$满足$\Gamma(0)=p$，初始切向量$\Gamma^{'}(0)=\mathbf{v}$，其中$p \in \mathcal{M,\mathbf{v} \in \mathcal{T}_p \mathcal{M}}$，则在p处的指数映射定义为：
$$
Exp_p(v)=\Gamma(1)
$$
指数映射和对数映射是流形和它的切空间之间的**同构映射**，计算这些映射的算法依赖于感兴趣的流形和切空间的基点。

> 设 $V$与 $V^{\prime}$ 都是域 $F$ 上的线性空间，如果存在 $V$ 到 $V^{^{\prime}}$的一个双射 $\sigma$ ,并且 $\sigma$ 保持加法和数乘封闭，即
>
$$
\begin{array}{c}\sigma(\alpha+\beta)=\sigma(\alpha)+\sigma(\beta)\\\sigma(k\alpha)=k\sigma(\alpha)\end{array}
$$
>
>则称 $\sigma$ 是 $V$ 到 $V^{^{\prime}}$ 的同构映射(简称为同构),此时称 $V$ 与 $V^{^{\prime}}$ 是同构的，记 $V\cong V^{^{\prime}}$

- 给定SPD矩阵 $\mathcal{M} \in\mathcal{S}_{++}^{d}$，对应的**矩阵对数函数**为 $\mathrm{logm}({M}):\mathcal{S}_{++}^d \rightarrow sym(d)$ ：

$$
\operatorname{logm}(M)=U\log(\Sigma)U^\top 
$$

sym(d)代表dxd的对称矩阵张成的空间，${U\Sigma U}^{\top}=M$

- 给定对称矩阵 $N \in sym(d)$，其中 ${U\Sigma U}^{\top}=\mathbf{N}$，**矩阵指数函数**为 $expm(\mathbf{N}:sym(d)\rightarrow \mathcal{S}_{++}^d)$：

$$
\operatorname{expm}({N})={U}\exp({\Sigma}){U}^\top 
$$

> 存在另外一种指数对数的计算公式
> 
$$
\begin{aligned}\operatorname{Exp}_P(Q)&=P^{\frac12}\exp(P^{\frac12}QP^{-\frac12})P^{\frac12}\in S_d^+,\\ \operatorname{Log}_Q(P)&=Q^{\frac12}\log(Q^{\frac12}PQ^{-\frac12})Q^{\frac12}\in S_d.\end{aligned}
$$

通过映射到Cholesky空间，得到了黎曼指数映射和对数映射的易于计算的表达式, $X\in\mathcal{L}_+\text{ and }K\in\mathcal{L}$：

$$
\begin{aligned}\widetilde{\operatorname{Exp}}_X(K)&=\lfloor X\rfloor+\lfloor K\rfloor+\mathcal{D}(X)\exp\{\mathcal{D}(K)\mathcal{D}(X)^{-1}\},\\\\\widetilde{\operatorname{Log}}_K(X)&=\lfloor X\rfloor-\lfloor K\rfloor+\mathcal{D}(K)\log\{\mathcal{D}(K)^{-1}\mathcal{D}(X)\}.\end{aligned}
$$



### Frechet 平均
见前文

不需要迭代操作的对数乔列斯基平均 $\mu_{\mathcal{L}_+}$ ，其定义为：

$$
\mu_{\mathcal{L}_+}=\frac{1}{N}\sum_{i=1}^{N}\lfloor X_i\rfloor+\exp\left\{N^{-1}\sum_{i=1}^{N}\log\mathcal{D}(X_i)\right\}.
$$


### 流形ODE

假设z(T)是在M中取值的可微曲线，M是流形空间，动态系统的向量场f是一个 $\mathcal{C}^{1}$ 映射，使得

$$
\frac{d\mathbf{z}(t)}{dt}=f(\mathbf{z}(t),t)\in\mathcal{T}_{\mathbf{z}(t)}\mathcal{M}.
$$

向量场f称为流形上的微分方程,函数z称为积分曲线或简单地称为方程的解

## Challenges
### 基于实数域的神经网络的方法存在以下问题
- (I)在处理非欧特性的数据方面不足，例如，在流形空间中处理数据固有的非线性特征和几何特征
- (II)用于处理复杂和高维数据的数据不足,在训练时会扭曲几何结构而产生数值误差


### SPD矩阵具有rigid constraints
欧氏度量中的SPD矩阵会导致不利的结果，如膨胀效应和具有非正特征值的对称矩阵的有限距离[2]；此外，优化过程中的反向传播使得深度神经网络中的操作数分量很难保持正定性


## Solutions
提出了一种基于时序流形数据的连续流形学习方法，利用黎曼流形和Cholesky空间之间的微分同胚映射[2]，可以有效地解决最优化问题，大大降低计算成本；开发一种表示时间序列数据中固有的时间动力学的流形ODE，使其能够学习具有序列流形数据几何特征的轨迹

### 映射到黎曼Cholesky空间

提取特征后，利用shrinkage estimator 得到特征的二阶统计量“协方差”，即表示流形空间的SPD矩阵S；

S被分解为 $C(S)=XX^{T}$ ,X为下三角矩阵，满足 $X= \lfloor X \rfloor + \mathcal{D}(X)$

在Cholesky空间中进行运算时，由于Cholesky矩阵的良好性质，只考虑对角矩阵D（X）的元素为正的约束，而X中的元素是无约束的。

### Cholesky空间的循环网络 Riemannian manifold GRU

#### 在Cholesky空间重新定义门操作

$$
\begin{cases}\mathbf{z}_i&=\sigma(\mathrm{wFM}(\left\{X_i,H_{i-1}\right\},W_z)\oplus B_z),\\\mathbf{r}_i&=\sigma(\mathrm{wFM}(\left\{X_i,H_{i-1}\right\},W_r)\oplus B_r),\\\mathbf{l}_i&=\mathrm{wFM}(\left\{X_i,\mathbf{r}_i\odot H_{i-1}\right\},W_l)\oplus B_l,\\\hat{H}_i&=\tanh(\left\lfloor\mathbf{l}_i\right\rfloor)+\mathrm{softplus}(\mathcal{D}(\mathbf{l}_i)),\\H_i&=(1-\mathbf{z}_i)\odot H_{i-1}+\mathbf{z}_i\odot\hat{H}_i,&\end{cases}
$$

其中 

$$
\begin{gathered}
\begin{aligned}\text{wFM}(\{X_i\}_{i=1,\dots,N},\textbf{w}\in\mathbb{R}_{\geq0}^N)\end{aligned} \\
=\frac1N\sum_{i=1}^N(w_i\cdot\lfloor X_i\rfloor)+\exp\left\{N^{-1}\sum_{i=1}^Nw_i\cdot\log\mathcal{D}(X_i)\right\} 
\end{gathered}
$$

$$
X\oplus Y=\lfloor X\rfloor+\lfloor Y\rfloor+\mathcal{D}(X)\mathcal{D}(Y).
$$

### 神经流形ODE[3]
在获取序列数据用神经网路f定义动态模型，定义了网络参数学习的前向和后向传递梯度计算

#### 前向过程 （流形上计算）
流形上ODE的传播方式有两种：
- 映射方法[4]
- 隐式方法[4] [5] [7]

#### 后向传播 （欧氏空间上的ODE）
[3] 和 [7] 提出了一种有效地独立计算流形方程的**梯度**和**导数**的伴随灵敏度方法

在[3]中，定义损失函数 $E:\mathcal{M} \rightarrow \mathbb{R}$ ，伴随状态 $a(t)=D_{z{t}}E$ 满足 $\frac{da(t)}{dt}=-a(t)D_z(t)f_{\Phi}(z(t),t)$，有微分方程：

$$
\frac{d \widetilde{Log}(H_t)}{dt}=D_{\widetilde{Exp}(H_t)}\widetilde{Log}(f_{\Phi}(\widetilde{Exp}(H_t),t))
$$

用数值积分技术求解 $\widetilde{Log}(H_t)$ ，以 $\epsilon$ 更新 $H_t$ ，结合起始时间和初始条件权重得到梯度（详见 [3]）

## Reference
[1] S. Jeong, W. Ko, A. W. Mulyadi, and H.-I. Suk, “Deep Efficient Continuous Manifold Learning for Time Series Modeling,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 1, pp. 171–184, Jan. 2024, doi: 10.1109/TPAMI.2023.3320125.

[2] Z. Lin, “Riemannian Geometry of Symmetric Positive Definite Matrices via Cholesky Decomposition,” SIAM J. Matrix Anal. Appl., vol. 40, no. 4, pp. 1353–1370, Jan. 2019, doi: 10.1137/18M1221084.

[3]A. Lou et al., “Neural manifold ordinary differential equations,” in Proc. Adv. Neural Inf. Process. Syst., 2020, pp. 17548–17558.

[4] E. Hairer, “Solving ordinary differential equations on manifolds,” in Lecture Notes, University of Geneva, 2011.

[5]P. E. Crouch and R. Grossman, “Numerical integration of ordinary differential equations on manifolds,” J. Nonlinear Sci., vol. 3, no. 1, pp. 1–33, 1993.

[6] A. Bielecki, “Estimation of the Euler method error on a Riemannian manifold,” Commun. Numer. Methods Eng., vol. 18, no. 11, pp. 757–763, 2002

[7] R. T. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud, “Neural ordinary differential equations,” in Proc. Adv. Neural Inf. Process. Syst., 2018, pp. 6572–6583.

# 3.manifold-constrained Gaussian Process for dynamic system

## Scenes and Background
在有噪声和稀疏数据情况下，ODE表示的非线性系统的参数估计

> In deep learning and manifold learning, a manifold constraint refers to a constraint on the parameters of a model that restricts them to lie on a smooth Riemannian manifold 1. Manifold optimization is a nonlinear optimization problem that translates the constrained optimization problem into an unconstrained optimization over the manifold, thereby generalizing many of the standard nonlinear optimization algorithms with guarantees 1.Such constraints include the popular orthogonality and rank constraints, and have been recently used in a number of applications in deep learning 1.

## Challenges
- 解数值方程需要进行积分计算以及大量的时间
- dynamic系统中可能存在未被观察的数据



## Solutions
针对时间数据进行高斯过程建模，模型满足高斯约束：即高斯过程的导数必须满足微分方程组的动力学性质。

对于一个解 $X(t)$ ，如果其先验是确定的，那么其导数的条件分布也是确定的，这与实际中普适成立的微分方程存在矛盾

为解决上述矛盾问题，定义随机变量 $W$ 满足 $W=\sup_{t\in[0,T],d\in\{1,...,D\}}|\dot{X}_d(t)-\mathbf{f}(\bfsymbol{X}(t),\bfsymbol{\theta},t)_d|$ ，当 $W \equiv 0$ 即说明随机过程 $X(t)$ 与ODE一致。那么对于 $X(t)$ 的后验概率分布为：

$$
p_{\bfsymbol{\Theta},\bfsymbol{X}(t)|W,\bfsymbol{Y}(\bfsymbol{\tau})}(\bfsymbol{\theta},\bfsymbol{x}(t)|W=0,\bfsymbol{Y}(\bfsymbol{\tau})=\bfsymbol{y}(\bfsymbol{\tau}))
$$

为便于计算，将 $W$ 进行离散化，即 $W_I=\max_{t\in\bfsymbol{I},d\in\{1,...,D\}}|\dot{X}_d(t)-\mathbf{f}(\bfsymbol{X}(t),\bfsymbol{\theta},t)_d|$ ，其中 $\bfsymbol{I}=(t_1,t_2,\ldots,t_n)$ ，最终得到：

$$
\begin{aligned}
&p_{\bfsymbol{\Theta},\bfsymbol{X}(\bfsymbol{I})|\bfsymbol{W}_I,\bfsymbol{Y}(\bfsymbol{\tau})}(\bfsymbol{\theta},\bfsymbol{x}(\bfsymbol{I})|\bfsymbol{W}_I=0,\bfsymbol{Y}(\bfsymbol{\tau})=\bfsymbol{y}(\bfsymbol{\tau}))&   \\
&\propto\pi_\Theta(\bfsymbol{\theta})\exp\left\{-\frac12\sum_{d=1}^D\right[ \\
&+\underbrace{|\bfsymbol{I}|\log(2\pi)+\log|C_d|+\|x_d(\bfsymbol{I})-\mu_d(\bfsymbol{I})\|_{C_d^{-1}}^2}_{(1)} \\
&\underbrace{+|\bfsymbol{I}|\log(2\pi)+\log|K_d|+\left\|\mathbf{f}_{d,\bfsymbol{I}}^{\mathbf{x},\bfsymbol{\theta}}-\dot{\mu}_d(\bfsymbol{I})-m_d\{x_d(\bfsymbol{I})-\mu_d(\bfsymbol{I})\}\right\|_{K_d^{-1}}^2}_{(3)} \\
&\left.\left.+\underbrace{N_d\log(2\pi\sigma_d^2)+\left\|x_d(\tau_d)-y_d(\tau_d)\right\|_{\sigma\frac dd^2}^2}_{(2)}\right]\right\},
\end{aligned}
$$

# 4.manifold-regularized model for prediction improvement (Manifold alignment)


## Scenes and Background
在多尺度机制中，多模态数据构造的复杂数据能够加深对于复杂机制的理解；


## Challenges
- 集成和分析这样的多模态数据仍然具有挑战性，这些数据通常是**高维的**和**异质的**
- 跨模态数据间的非线性关系常常在预测中被忽略



## Solutions
开发出一种可解释的正则化模型，从多模态数据中进行预测
- 1.DNN学习跨模态流形，**对齐**（align）流形特征[3]
> 目的是保持各模间的全局一致性和局部光滑性，并揭示高阶非线性跨模间关系

- 2.使用跨模态流形作为feature graph来**正则化**分类器来提升预测精度[2]
> 相似特征在训练后应该具有相似的权重，因此我们通过每个特征与其相邻特征的加权平均的差值的平方来规则化每个特征的权重

- 3.设计了一个优化算法来反向传播**Stiefel**流形上的**黎曼梯度**[4]
> The projections onto the Stiefel manifold and the Euclidean gradient onto the tangent space of the Stiefel manifold are illustrated in Supplementary Fig. 6.

## Reference
[1] N. D. Nguyen, J. Huang, and D. Wang, “A deep manifold-regularized learning model for improving phenotype prediction from multi-modal data,” Nat Comput Sci, vol. 2, no. 1, pp. 38–46, Jan. 2022
[2] Sandler, T., Blitzer, J., Talukdar, P. & Ungar, L. Regularized learning with networks of features. Adv. Neural Inf. Process. Syst. 21, 1401–1408 (2008).
[3] Wang, C., Krafft, P., Mahadevan, S., Ma, Y. & Fu, Y. Manifold alignment. In Manifold Learning: Theory and Applications 95–120 (CRC, 2011).
[4] Stiefel, E. Richtungsfelder und Fernparallelismus in n-dimensionalen Mannigfaltigkeiten. Commentarii Math. Helvetici 8, 305–353 (1935).


# 5.Matern Gaussian process on Riemannian manifolds

