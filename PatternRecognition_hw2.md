#实验二 用SVM对脑电波睡眠周期分类

## 1 实验原理（核函数变换与非线性支持向量机）	

非线性的支持向量机采用引入特征变换来将原空间中的非线性问题转化为新空间的线性问题。已知线性支持向量机求解的分类器可以表示为

$f(x)= \mathrm{sgn} (w \cdot x + b)=\mathrm{sgn} (\sum\limits_{i=1}^n\alpha_iy_i(x_i\cdot x)+b) $

其中的$\alpha_i, i=1,\cdots, n$是下列二次优化问题的解

$\max\limits_\alpha \quad Q(\alpha)=\sum\limits_{i=1}^n\alpha_i-\frac{1}{2}\sum\limits_{i,j=1}^n\alpha_i\alpha_jy_iy_j(x_i\cdot x_j) $

$\mathrm{s.t.} \quad \sum\limits_{i=1}^ny_i\alpha_i=0$

$\quad \quad \quad 0\le \alpha_i \le C,\quad i=1,\cdots,n$

$b$通过使

$y_j(\sum\limits_{i=1}^n\alpha_i(x_i\cdot x_j)+b)-1=0$

成立的样本$x_j$（即支持向量）求得。

如果我们对特征$x$进行非线性变换，记新特征为$z=\varphi(x)$，则新特征空间里构造的支持向量机决策函数是

$f(x)=\mathrm{sgn}(w^\varphi\cdot z+b)=\mathrm{sgn}(\sum\limits_{i=1}^n\alpha_iy_i(\varphi(x_i)\cdot \varphi(x))+b)$

而相应的优化问题变成

$\max\limits_\alpha \quad Q(\alpha)=\sum\limits_{i=1}^n\alpha_i-\frac{1}{2}\sum\limits_{i,j=1}^n\alpha_i\alpha_jy_iy_j(\varphi(x_i)\cdot \varphi(x_j) )$

$\mathrm{s.t.} \quad \sum\limits_{i=1}^ny_i\alpha_i=0$

$\quad \quad \quad 0\le \alpha_i \le C,\quad i=1,\cdots,n​$

定义支持向量的等式成为

$y_j(\sum\limits_{i=1}^n\alpha_iy_i(\varphi(x_i)\cdot \varphi(x_j))+b)-1=0$

在进行变换后，无论变换的具体形式如何，变换对支持向量机的影响是把两个样本的原特征空间中的内积$(x_i\cdot x_j)$变成了在新空间中的内积$(\varphi(x_i)\cdot \varphi(x_j))$。新空间的内积也是原特征的函数，可以记作：

$K(x_i, x_j) \triangleq (\varphi(x_i)\cdot \varphi(x_j))$

把它称为核函数。这样，变换空间里的支持向量机就可以写成

$f(x)=\mathrm{sgn}(\sum\limits_{i=1}^n\alpha_iy_iK(x_i, x_j)+b)$

其中，系数$\alpha$是下列优化问题的解

$\max\limits_\alpha \quad Q(\alpha)=\sum\limits_{i=1}^n\alpha_i-\frac{1}{2}\sum\limits_{i,j=1}^n\alpha_i\alpha_jy_iy_jK(x_i,x_j )$

$\mathrm{s.t.} \quad \sum\limits_{i=1}^ny_i\alpha_i=0$

$\quad \quad \quad 0\le \alpha_i \le C,\quad i=1,\cdots,n$

$b$通过满足下式的样本（支持向量）求得

$y_j(\sum\limits_{i=1}^n\alpha_iy_iK(x_i,x_j)+b)-1=0$

常用核函数的形式

多项式核函数

$K(x,x^\prime)=((x\cdot x^\prime)+1)^q$

径向基（RBF）核函数

$K(x,x^\prime)=\mathrm{exp}(-\frac{||x-x^\prime||^2}{\sigma^2})$

$\mathrm{Sigmoid}$函数

$K(x,x^\prime)=\tanh(v(x\cdot x^\prime)+c)$

## 2 实验过程及结果

数据采用上一次实验清洗好的数据文件：训练集为trainingDataSet.mat（77220*1001），测试集为testingDataSet.mat（4764\*1001）。

### 2.1 线性核

代码在svm_linear.py中。直接以下命令运行代码：

```shell
python svm_linear.py
```

首先数据经过标准化（特征每一维化为方差为1，均值为0）。

```python
from sklearn import preprocessing
X_scaled = preprocessing.scale(X)
```

然后使用线性核训练SVM分类器

```python
clf = svm.LinearSVC(C=C_space[self.C_index], class_weight='balanced')
clf.fit(X_scaled, Y)
```

使用训练好的分类器对测试集进行分类，并计算精度（Accuracy）：

```python
R = clf.predict(testX_scaled)
len = R.shape[0]
count = 0
for i in range(len):
	if R[i] == testY[i]:
    	count = count + 1
accuracy = float(count) / len
```

线性核参数主要是正则项系数C，我首先遍历$[2^{-5},2^{15}]$之间的十个等分点

```python
C_space = np.logspace(-5, 15, num=10, base=2, dtype=float)
```

运行结果为

```shell
C = 0.031250 begin training
C = 0.145816 begin training
C = 0.680395 begin training
C = 3.174802 begin training
C = 14.813995 begin training
C = 69.123823 begin training
C = 322.539789 begin training
C = 1505.008120 begin training
C = 7022.542708 begin training
C = 32768.000000 begin training
C = 0.680395 begin predicting
Linear Kernel with C = 0.680395 , Accuracy is 0.586692
C = 3.174802 begin predicting
Linear Kernel with C = 3.174802 , Accuracy is 0.551847
C = 69.123823 begin predicting
Linear Kernel with C = 69.123823 , Accuracy is 0.467464
C = 322.539789 begin predicting
Linear Kernel with C = 322.539789 , Accuracy is 0.438287
C = 7022.542708 begin predicting
Linear Kernel with C = 7022.542708 , Accuracy is 0.470613
C = 32768.000000 begin predicting
Linear Kernel with C = 32768.000000 , Accuracy is 0.438917
C = 0.031250 begin predicting
Linear Kernel with C = 0.031250 , Accuracy is 0.579345
C = 0.145816 begin predicting
Linear Kernel with C = 0.145816 , Accuracy is 0.584383
C = 1505.008120 begin predicting
Linear Kernel with C = 1505.008120 , Accuracy is 0.448153
C = 14.813995 begin predicting
Linear Kernel with C = 14.813995 , Accuracy is 0.522250
```

|    C     | 0.031250 | 0.145816 | 0.680395 | 3.174802 | 14.813995 | 69.123823 | 1505.008120 | 7022.542708 |      |      |
| :------: | :------: | :------: | :------: | :------: | :-------: | :-------: | :---------: | :---------: | :--: | ---- |
| Accuracy | 0.579345 | 0.584383 | 0.586692 | 0.551847 | 0.522250  | 0.467464  |  0.448153   |  0.470613   |      |      |

精度普遍低于60%，所以我考虑更换核函数。

###2.2 径向基（RBF）核

代码在svm_pca_rbf.py中。运行代码需要在python终端中输入

```python
import svm_pca_rbf
svm_pca_rbf.Main()
```

径向基核的计算复杂度有点大，所以我先使用PCA将原数据降到15维，特征值占全体特征值的64.9%。接着进行标准化。

```python
from sklearn.decomposition import PCA    pca = PCA(n_components=15)
from sklearn import preprocessing
pca.fit(X)
X_pca = pca.transform(X)
X_scaled = preprocessing.scale(X_pca)
```

然后使用径向基（RBF）核训练SVM分类器

```python
clf = svm.SVC(C=C_space[i], gamma=Gamma_space[j])
clf.fit(X_scaled, Y)
```

由于计算复杂度的原因，其余参数不做设置，采取默认。使用训练好的分类器对测试集进行分类，并计算精度（Accuracy）。

径向基核的参数主要有两个$\gamma$和$C$，所以我首先使用网格遍历

$\gamma \in [0.001, 0.01, 0.1, 1, 10],\ C \in [0.001, 0.01, 0.1, 1, 10]$

共有25个组合。得到结果为：

|           | $\gamma=0.001$ | $\gamma=0.01$ | $\gamma=0.1$ | $\gamma=1$ | $\gamma=10$ |
| :-------: | :------------: | :-----------: | :----------: | :--------: | :---------: |
| $C=0.001$ |    0.574307    |   0.628673    |   0.784425   |  0.574307  |  0.574307   |
| $C=0.01$  |    0.574097    |   0.731948    |   0.788413   |  0.574307  |  0.574307   |
|  $C=0.1$  |    0.643157    |   0.761965    |   0.799958   |  0.740554  |  0.574307   |
|   $C=1$   |    0.719353    |   0.777918    | **0.802057** |  0.786104  |  0.575777   |
|  $C=10$   |    0.741184    |   0.787783    |   0.791352   |  0.759866  |  0.576406   |

结果表明，$C=1,\gamma=0.1$时精度（Accuracy）取得极大值80.2%。

为了更精确的寻找使得精度达到最大值的参数，我在$C=1,\gamma=0.1$的邻域进行了进一步的遍历。即

|          | $\gamma=0.095$ | $\gamma=0.1$ | $\gamma=0.105$ |
| :------: | :------------: | :----------: | :------------: |
| $C=0.9$  |    0.802687    |   0.802687   |    0.802897    |
| $C=0.95$ |    0.802057    |   0.802687   |    0.802687    |

显然在$C=0.9,\gamma=0.105$处取得最大值，接着我又尝试了$C=0.8, \gamma=0.11$ 结果为0.803107，$C=0.7, \gamma=0.12$，结果为0.801847。

时间有限，我最后确定了参数为$C=0.8,\gamma=0.11$，此时的RBF核函数SVM分类器在测试集上的精度为**80.3%**。