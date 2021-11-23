## **Tensor的操作**

张量的数据属性与 NumPy 数组类似，如下所示：

![微信图片_20210723234005](https://cdn.jsdelivr.net/gh/youminglan/Picture@main/img/20210723234033.jpg)

张量的操作主要包括张量的结构操作和张量的数学运算操作。

- Tensor的结构操作包括：**创建张量**，**查看属性**，**修改形状**，**指定设备**，**数据转换**， **索引切片**，**广播机制**，**元素操作**，**归并操作**；
- Tensor的数学运算包括：**标量运算**，**向量运算**，**矩阵操作**，**比较操作**。

### 创建张量

Pytorch中创建张量的方法有很多，如下图所示：

![微信图片_20210726203204](https://cdn.jsdelivr.net/gh/youminglan/Picture@main/img/20210726203242.jpg)

在深度学习过程中最多使用$5$个维度的张量：**标量**（0维张量）**，向量**（1维度张量）**，矩阵**（2维张量），**3维张量，4维张量，5维张量**。

#### 创建标量**（0维张量）**

仔细观察下述代码，看看有什么区别：

```python
x = torch.tensor(2)
print(x, x.shape, x.type())
y = torch.Tensor(2)
print(y, y.shape, y.type())
```

```python
tensor(2) torch.Size([]) 
torch.LongTensortensor([0., 0.]) torch.Size([2]) 
torch.FloatTensor
```

注意到了torch.tensor与torch.Tensor的区别没？一字之差，结果差别却很大。

- torch.Tensor(2) 使用全局默认 dtype（FloatTensor），返回一个size为$2$的向量，初值为 $0$；
- torch.tensor(2) 返回常量 $2$，数据类型从数据推断而来，其中的$2$表示的是数据值。

#### **创建向量**（1维度张量）

向量只不过是一个元素序列的数组。例如，表示一个地区一段时间的气温。

```python
x = torch.FloatTensor([23.5, 24.6, 25.9, 26.1])
print(x, x.shape, x.type())
```

```python
tensor([23.5000, 24.6000, 25.9000, 26.1000]) torch.Size([4]) 
torch.FloatTensor
```

#### **创建矩阵（2维向量）**

从上面可知，创建矩阵的方式有很多，我们选择 from_numpy 的方式将 numpy 数组转换成 torch 张量。下面以**波士顿房价的数据集**为例子，它包含在机器学习包scikit-learn中。该数据集包含了 506 个样本，其中每个样本有 13 个特征。

```python
from sklearn.datasets import load_bostonboston = load_boston() # 下载数据集
boston_tensor=torch.from_numpy(boston.data)
print(boston_tensor[:2])
print(boston_tensor.shape)
print(boston_tensor.type())
```

```python
tensor([[6.3200e-03, 1.8000e+01, 2.3100e+00, 0.0000e+00, 5.3800e-01, 6.5750e+00,     6.5200e+01, 4.0900e+00, 1.0000e+00, 2.9600e+02, 1.5300e+01, 3.9690e+02,     4.9800e+00],    [2.7310e-02, 0.0000e+00, 7.0700e+00, 0.0000e+00, 4.6900e-01, 6.4210e+00,     7.8900e+01, 4.9671e+00, 2.0000e+00, 2.4200e+02, 1.7800e+01, 3.9690e+02,     9.1400e+00]], dtype=torch.float64)
torch.Size([506, 13])
torch.DoubleTensor
```

最常见的三维张量就是图片，例如$[224, 224, 3]$，下面我们演示如何加载图片数据。

```python
from PIL import Image
panda = np.array(Image.open("../images/panda.jpg").resize((224,224)))
panda_tensor=torch.from_numpy(panda)
print(panda_tensor.size())
print(panda_tensor.dtype)
plt.imshow(panda_tensor)
```

```python
torch.Size([224, 224, 3])torch.uint8
```

![image-20210726203729720](https://cdn.jsdelivr.net/gh/youminglan/Picture@main/img/20210726203729.png)

#### **创建4维张量**

4维张量最常见的例子就是批图像。例如，加载一批 $[64, 224, 224, 3]$ 的图片，其中 $64$ 表示批尺寸，$[224, 224, 3]$ 表示图片的尺寸。

```python
from glob import glob
data_path = "./data/cats/"
imgs = glob(data_path+'*.jpg')
imgs_np = np.array([np.array(Image.open(img).resize((224,224))) for img in imgs])
imgs_np = imgs_np.reshape(-1, 224, 224, 3)
imgs_tensor=torch.from_numpy(imgs_np)
print(imgs_tensor.shape)
print(imgs_tensor.dtype)
```

```python
torch.Size([397, 224, 224, 3])
torch.uint8
```

上面代码中一共读取了 $397$ 张图片。

#### **创建5维张量**

使用5维度张量的例子是视频数据。视频数据可以划分为片段，一个片段又包含很多张图片。例如，$[32, 30, 224, 224, 3]$ 表示有 $32$ 个视频片段，每个视频片段包含 $30$ 张图片，每张图片的尺寸为 $[224, 224, 3]$。下面，我们模拟产生这样一个尺寸的5维数据（注意：只是模拟产生$5$维数据，并不是真的视频数据）。

```python
video_tensor=torch.randn(32,30,224,224,3)
print(video_tensor.shape)
print(video_tensor.dtype)
```

```python
torch.Size([32, 30, 224, 224, 3])
torch.float32
```



### **查看属性**

张量有很多属性，下面我们看看常用的属性有哪些？

- tensor.shape，tensor.size(): 返回张量的形状；
- tensor.ndim：查看张量的维度；
- tensor.dtype，tensor.type()：查看张量的数据类型；
- tensor.is_cuda：查看张量是否在GPU上；
- tensor.grad：查看张量的梯度；
- tensor.requires_grad：查看张量是否可微。

```python
tensor = torch.randn(2,3)
print("形状: ", tensor.shape, tensor.size())
print("维度: ", tensor.ndim)
print("类型: ", tensor.dtype, tensor.type())
print("cuda: ", tensor.is_cuda)
print("梯度: ", tensor.grad)
```

```python
形状:  torch.Size([2, 3]) torch.Size([2, 3])
维度:  2
类型:  torch.float32 torch.FloatTensor
cuda:  False    
梯度:  None
```

其中，torch.FloatTensor 就是$32$位的浮点数。



### **修改张量的形状**

在处理数据和构建网络的时候，时常需要修改 Tensor 的形状。涉及到修改形状的常见函数如下：

- tensor.numel()：计算Tensor的元素个数；
- tensor.view(*shape)：修改Tensor的形状。view()返回的Tensor与源Tensor共享内容。使用view必须要求源Tensor是连续的，否则会执行失败。view(-1)实现展平；
- tensor.resize(*shape)：功能类似与 view，resize不要求Tensor内存连续；
- tensor.reshape(*shape)：修改Tensor的形状，Reshape返回新的Tensor；
- tensor.unsqueeze(pos)：在指定位置添加一个维度；
- tensor.squeeze()：消除维度为$1$的维。

```python
import torch
x = torch.randn(2,3)
print("元素个数: {}".format(x.numel()))
# view调整尺寸
print("\ntensor.view(3,2): \n{}".format(x.view(3,2)))
print("tensor.view(-1): {}".format(x.view(-1)))
# resize调整尺寸
print("\ntensor.resize(3,2): \n{}".format(x.resize(3,2)))
# reshape调整尺寸
print("\ntensor.reshape(3,2): \n{}".format(x.reshape(3,2)))
print("tensor.reshape(-1): {}".format(x.reshape(-1)))
#添加一个维度
x123 = x.unsqueeze(0)
x213 = x.unsqueeze(1)
x231 = x.unsqueeze(2)
print("\ntensor.unsqueeze(0): {}".format(x123.shape))
print("tensor.unsqueeze(1): {}".format(x213.shape))
print("tensor.unsqueeze(2): {}".format(x231.shape))
# 去掉维度为1的维
print("\ntensor.squeeze(): {}".format(x123.squeeze().shape))
print("tensor.squeeze(): {}".format(x213.squeeze().shape))
print("tensor.squeeze(): {}".format(x231.squeeze().shape))
```



```python
元素个数: 6
tensor.view(3,2): 
tensor([[ 0.1274, -1.5990],        [-0.8852, -1.3436],        [-0.7716,  1.5765]])
tensor.view(-1): tensor([ 0.1274, -1.5990, -0.8852, -1.3436, -0.7716,  1.5765])
tensor.resize(3,2): 
tensor([[ 0.1274, -1.5990],        [-0.8852, -1.3436],        [-0.7716,  1.5765]])
tensor.reshape(3,2): 
tensor([[ 0.1274, -1.5990],        [-0.8852, -1.3436],        [-0.7716,  1.5765]])
tensor.reshape(-1): 
tensor([ 0.1274, -1.5990, -0.8852, -1.3436, -0.7716,  1.5765])
tensor.unsqueeze(0): torch.Size([1, 2, 3])
tensor.unsqueeze(1): torch.Size([2, 1, 3])
tensor.unsqueeze(2): torch.Size([2, 3, 1])
tensor.squeeze(): torch.Size([2, 3])
tensor.squeeze(): torch.Size([2, 3])
tensor.squeeze(): torch.Size([2, 3])
```



在上述函数中，有三个函数都可以调整Tensor的尺寸：view，reshape，resize。他们之间有什么区别嘛？

- reshape可以由torch.reshape()，tensor.reshape()调用，而view只能通过tensor.view()调用。
- view()方法只能改变连续的张量，否则必须先调用.contiguous()方法使内存连续；.reshape()方法不受此限制。方法.transpose(), .permute()会使的Tensor在内存中不连续。
- view() 返回的Tensor与源Tensor共享内存；.reshape()返回的Tensor与源Tensor不共享内存；
- resize() 与 .reshape()效果类似。
- 如果只想重塑Tensor，建议使用.reshape；如果关注内存希望两个Tensor共享内存，建议使用.view()。



### 指定设备

PyTorch 为CPU 和 GPU 提供了不同的张量实现。每个张量都可以转化到 GPU 中，以便大规模计算。创建Tensor时，默认指定的设备是CPU。

创建Tensor时，我们可以通过torch.tensor([...],dtype=,device='cpu/cuda') 指定Tensor所属的设置是CPU还是GPU；也可以通过 tensor.to(device=cpu/cuda)或者tensor.cuda()，tensor.cpu() 把张量转化到指定的设备上。

```python
# 创建CPU上的张量
tensor_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64, device='cpu')
print(f"tensor_cpu = \n{tensor_cpu}")
# 创建GPU上的张量
tensor_gpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64, device='cuda')
print(f"tensor_gpu = \n{tensor_gpu}")
```

```python
tensor_cpu = tensor([[1., 2.],        [3., 4.],        [5., 6.]], dtype=torch.float64)
tensor_gpu = tensor([[1., 2.],        [3., 4.],        [5.,6.]],device='cuda:0',dtype=torch.float64)
```

```python
# cpu -> gpu
tensor_gpu_cpu = tensor_gpu.to(device='cpu')print(f"tensor_gpu_cpu = {tensor_gpu_cpu}")
# gpu -> cpu
tensor_cpu_gpu = tensor_cpu.to(device='cuda')print(f"tensor_cpu_gpu = {tensor_cpu_gpu}")
```

```python
tensor_gpu_cpu = tensor([[ 5., 10.],        [15., 20.],        [25., 30.]], dtype=torch.float64)
tensor_cpu_gpu = tensor([[ 5., 10.],        [15., 20.],        [25., 30.]], device='cuda:0', dtype=torch.float64)
```

### **数据转换**

有时候，我们需要把Tensor转换成普通的数据，我们可以使用下列方式进行操作：

- .numpy()：把Tensor转换成numpy数据;
- .item()：如果Tensor为单元素，则返回Python标量；
- .detach()：返回一个与当前计算图分离且无梯度的新张量；

```python
# 把tensor转化为numpy array
f = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
f_numpy = f.numpy()
print(type(f_numpy), f_numpy) # <class 'numpy.ndarray'> [1. 2. 3. 4.]
# 把单元素Tensor转化为python标量
f_item = f[0].item()
print(type(f_item), f_item) # <class 'float'> 1.0
# 获取分离计算图的新张量
a = torch.tensor([1.0, 2.0], requires_grad=True)
b = torch.tensor([3.0, 4.0], requires_grad=True)
c = a + bd = c.detach()
print("c.grad: {}; d.grad: {}".format(c.requires_grad, d.requires_grad))
# c is grad: True; d is grad: False
```



### **索引切片**

Tensor的索引切片操作与Numpy类似，也是Pytorch中经常使用的操作方式。需要注意的是，一般情况下索引结果与源数据共享内存。从Tensor中获取元素除了可以通过索引，也可以使用专有的函数。



常见的操作函数如下：



**★提取元素的操作：**

- torch.index_select(input, dim, index)：在指定维度上选择一些行或列；
- torch.nonzero(input)：获取非0元素的下标；
- torch.masked_select(input, mask)：使用二元值(真值表)进行选择元素；
- torch.gather(input, dim, index)：在指定的维度上选择数据，输出的形状与index一致；
- torch.take(input, index)：将输入看成一维数组，输出与index同形状。



**★会对元素进行修改的操作：**

- torch.scatter_(input, dim, index, src)：为gather的反操作，根据指定索引填充数据；
- torch.where(condition, x, y)：根据条件进行选择填充，这个操作函数用的非常多；
- torch.masked_fill：使用二元值(真值表)进行填充元素；
- torch.index_fill：使用下标进行填充。



代码从两部分进行演示，先展示提取元素的操作：

```python
# 索引操作
>>> x = torch.randn(2, 3)
tensor([[ 0.3607, -0.2859, -0.3938],        [ 0.2429, -1.3833, -2.3134]])
# index_select
>>> x = torch.randn(2, 3)
>>> torch.index_select(x, dim=1, index=torch.tensor([0,2])) tensor([[-0.5883,  0.4322],        [ 0.4612, -0.2675]])
# masked_select 
>>> x = torch.randn(2, 3)
>>> mask = x>0 # 产生真值Tensor
>>> select2 = torch.masked_select(x, mask) # 选择真值为True的元素
>>> select2
tensor([0.5605, 0.5895])
# nonzeros
>>> torch.nonzeros(mask)tensor([[0, 1],        [1, 0]])
# gather
>>> #out[i][j] = input[index[i][j]][j]  # if dim == 0
>>> #out[i][j] = input[i][index[i][j]]  # if dim == 1
>>> index = torch.LongTensor([[0, 1, 1]])
>>> gather1 = torch.gather(x, dim=0, index=index)
>>> print("gather1: {}".format(gather1))
gather1: tensor([[ 0.3607, -1.3833, -2.3134]])
    
>>> index = torch.LongTensor([[0,1,1],[1,1,1]])
>>> a = torch.gather(x, dim=1, index=index)
>>> print("gather2: {}".format(a))
gather2: tensor([[ 0.3607, -0.2859, -0.2859],        [-1.3833, -1.3833, -1.3833]])
# take
>>> src = torch.tensor([[4, 3, 5], [6, 7, 8]])
>>> out = torch.take(src, index=torch.tensor([0, 3, 5]))
>>> print("out: {}".format(out))
out: tensor([4, 6, 8])
```

会对元素进行修改的操作：

```python
# scatter_
>>> z = torch.zeros(2,3)
>>> b = z.scatter_(dim=1, index, a)
>>> print("b: {}".format(b))b: tensor([[ 0.3607, -0.2859,  0.0000],        [ 0.0000, -1.3833,  0.0000]])
# where
>>> x = torch.rand(3, 2)
>>> y = torch.ones(3, 2)>>> result = torch.where(x>0.5, x, y)
>>> print("result: {}".format(result))
result: tensor([[1.0000, 1.0000],        [1.0000, 0.8238],        [0.5557, 0.9770]])
# masked_fill
>>> x = torch.rand(3, 2)
>>> torch.masked_fill(x, x < 0.5, -1.0)
tensor([[-1.0000, -1.0000],        [ 0.8935, -1.0000],        [-1.0000, -1.0000]])
# index_fill
>>> x = torch.rand(3, 2)
>>> torch.index_fill(x , dim = 0, index = torch.tensor([0,1]), value = 100)
tensor([[100.0000, 100.0000],        [100.0000, 100.0000],        [  0.9798,   0.5548]])
```

在上述函数中，torch.gather() 和 torch.scatter_() 让人很难理解。借用官网的一段代码进行详细说明一下：

```python
# gather
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

>>> t = torch.tensor([[1, 2], [3, 4]])
>>> tor.gather(t, dim=1, index=torch.tensor([[0, 0], [1, 0]]))
tensor([[ 1,  1],        [ 4,  3]])
```

我们可以想象在index上进行填数，因为输出和index是一样的形状。如果dim=1，意为在index每个元素值作为j，该元素所在的行作为i，在t中进行取值填充；如果dim=0，意为在index 每个元素值作为i，该元素所在的列作为j，在t中进行取值填充。

```python
# scatter_
self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
>>> src = torch.arange(1, 11).reshape((2, 5))
>>> src
tensor([[ 1,  2,  3,  4,  5],        [ 6,  7,  8,  9, 10]])
>>> index = torch.tensor([[0, 1, 2, 0]])
>>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
tensor([[1, 0, 0, 4, 0],        [0, 2, 0, 0, 0],        [0, 0, 3, 0, 0]])
```

scatter_ 的作用与 gather 刚好相反，取值方式与 gather 也类似。



### **广播机制**

广播机制是向量运算的重要技巧。下面演示Tensor如何执行广播操作。torch.broadcast_tensors 可以将多个张量根据广播规则转换成相同的维度。

```python
>>> A = np.arange(0, 40, 10).reshape(4, 1)
>>> B = np.arange(0, 3)#把ndarray转换为Tensor
>>> A1 = torch.from_numpy(A)  #形状为4x1
>>> B1 = torch.from_numpy(B)  #形状为3
#Tensor自动实现广播
>>> C = A1 + B1 # 形状为4+3
tensor([[ 0,  1,  2],        [10, 11, 12],        [20, 21, 22],        [30, 31, 32]], dtype=torch.int32)
>>> A_broad, B_broad = torch.broadcast_tensors(A1, B1)
>>> C = A_broad + B_broad
tensor([[ 0,  1,  2],        [10, 11, 12],        [20, 21, 22],        [30, 31, 32]], dtype=torch.int32)
```



### **归并分割**

归并，意为对Tensor进行合并，这类操作的输出形状大于输入形状，是升维操作；分割，意为对Tensor进行切割细分，这类操作的输出形状小于输入形状，是降维操作。常见的操作函数如下：

- torch.cat(tensor, dim=0)：在指定维度连接多个Tensor，不会增加维度；
- torch.stack(tensor, dim=0)：在指定维度堆叠多个Tensor，会增加维度；
- torch.split(tensor, split, dim=0)：将一个张量分割为多个张量，不会减少维度，是torch.cat()的反向操作。

```python
# cat操作
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 0.6580, -1.0969, -0.4614],        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 0)
tensor([[ 0.6580, -1.0969, -0.4614],        [-0.1034, -0.5790,  0.1497],        [ 0.6580, -1.0969, -0.4614],        [-0.1034, -0.5790,  0.1497],        [ 0.6580, -1.0969, -0.4614],        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 1)
tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,         -1.0969, -0.4614],        [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,         -0.5790,  0.1497]])
# stack
>>> x = torch.randn(2, 3)
>>> torch.stack((x, x, x), dim = 0).size()
torch.Size([3, 2, 3])
>>> torch.stack((x, x, x), dim = 1).size()
torch.Size([2, 3, 3])
>>> torch.stack((x, x, x), dim = 2).size()
torch.Size([2, 3, 3])
# split
>>> a = torch.arange(10).reshape(5,2)
>>> atensor([[0, 1],        [2, 3],        [4, 5],        [6, 7],        [8, 9]])
>>> torch.split(a, 2)(tensor([[0, 1],         [2, 3]]), tensor([[4, 5],         [6, 7]]), tensor([[8, 9]]))
>>> torch.split(a, [1,4])(tensor([[0, 1]]), 
                          tensor([[2, 3],         [4, 5],         [6, 7],         [8, 9]]))
```



### **元素操作**

Tensor中也有很多元素操作的函数，也经常使用，例如：

- abs/add：绝对值和加法；
- ceil/floor：向上取整和向下取整；
- clamp(t, min, max)：将元素限定在指定区域内；
- round(t)：保留整数部分，四舍五入；
- trunc(t)：保留整数部分，向0归整；
- sigmoid/tanh/softmax：激活函数。

```python
>>> t = torch.randn(1, 3)
>>> torch.clamp(t, 0, 1)
tensor([[0., 0., 1.]])
```



###  **比较操作**

比较操作一般是进行逐个元素比较，常用的函数如下：

- eq：比较两个Tensor是否相等，支持广播操作；
- equal：比较两个Tensor是否具有相同的值和shape；
- max/min：返回最值；
- topk：返回指定维度上最高的K个值。

```python
>>> x=torch.linspace(0,10,6).view(2,3)
>>> torch.max(x) 
tensor(10.)
>>> torch.max(x,dim=0)
torch.return_types.max(values=tensor([ 6.,  8., 10.]),
                       indices=tensor([1, 1, 1]))
>>> torch.topk(x, k=1, dim=0)
torch.return_types.topk(
    values=tensor([[ 6.,  8., 10.]]),
    indices=tensor([[1, 1, 1]]))
```



### **标量运算**

Tensor的数学运算符分为：标量运算符，向量运算符，矩阵运算符，加减乘除，乘方，三角函数，指数，对数，逻辑比较运算等标量运算符。标量运算符的特点是逐元素运算。

```python
>>> a = torch.tensor([[1.0,2],[-3,4.0]])
>>> b = torch.tensor([[5.0,6],[7.0,8.0]])
>>> a+b # 加法
>>> a-b # 减法
>>> a*b # 乘法
>>> a/b # 除法
>>> torch.remainder(x, 2) # 取余数
>>> a**2 # 乘方
>>> a**(0.5) # 开方
>>> torch.sqrt(a) # 开方
>>> a%3 # 求模
>>> torch.fmod(a, 2)
>>> a//3 # 地板除法
>>> a >= 2 # torch.ge(a, 2)
>>> (a >= 2)&(a <= 3) # 逻辑运算
>>> (a >= 2)|(a <= 3) # 逻辑运算
```



### **向量运算**

向量运算符只在一个特定轴上运算，将一个向量映射到一个标量或者另外一个向量。

```python
# 统计值
a=torch.arange(1,10).float()
print(torch.sum(a))
print(torch.mean(a))
print(torch.max(a))
print(torch.min(a))
print(torch.prod(a))#累乘
print(torch.std(a))#标准差
print(torch.var(a))#方差
print(torch.median(a))#中位数
```



### **矩阵操作**

深度学习中存在大量的矩阵运算，常见的算法有两种：**逐个元素相乘**和**点积相乘**。Pytorch中常用的矩阵函数如下：

- dot(t1, t2)：计算Tensor（1D）的内积或者点积；
- mm(mat1,mat2)/bmm(batch1, batch2)：计算矩阵乘法/含batch的3D矩阵的乘法
- mv(t1, v1)：计算矩阵与向量的乘法；
- t：转置；
- svd(t)：计算t的SVD分解。

```python
printf("hello world!");
```



完结