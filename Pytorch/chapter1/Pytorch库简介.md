### PyTorch库概览



下图描述了一个典型的PyTorch工作流以及与每个步骤相关联的重要模块。

![图片](https://cdn.jsdelivr.net/gh/youminglan/Picture@main/img/20210723233529.webp)



我们将简单的讨论一下PyTorch的重要模块：torch.nn，torch.optim，torch.utils，torch.autograd。

#### 数据加载和处理

任何深度学习项目的第一步就是**加载并处理数据**。Pytorch通过torch.utils.data提供了该功能。

在这个模块中有两个很重要的类：Dataset 和 DataLoader。

- **Dataset**：是建立的张量的基础之上，主要用于创建自定义数据集。它的结构类似于列表，能够根据索引获取数据集的元素。PyTorch同时提供了常用的读取数据集更简单的接口，例如：torchvision.datasets.ImageFolder() 专门用来读取图片数据集。
- **DataLoader**：意为数据加载器，它定义了按batch加载数据的方法。它以迭代的方式批量的输出batch个数据。

#### **构建神经网络**

**torch.nn** 模块用于创建神经网络模块。它提供了所有常见的神经网络层：全连接层，卷积层，池化层，激活函数，损失函数等。、

一旦网络结构搭建完成，数据准备好输入网络，在训练模型的过程中，我们需要计算权重和偏置的梯度以更新权重参数和偏置参数。这些需求可以使用模块 **torch.optim** 模块完成。类似的，使用模块 **torch.autograd** 完成后向传递中的自动微分。

#### **模型推理**


在模型训练完成之后，可以用测试样例进行预测计算，这称之为**推理**。**模型推理**跟**模型训练**类似，只是不需要反馈计算。

你还可以将PyTorch训练的模型转换为**ONNX**格式，ONNX允许你在其他DL框架(如MXNet、CNTK、Caffe2)中使用这些模型。你也可以将**onnx**模型转换为Tensorflow。