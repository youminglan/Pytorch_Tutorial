### torchvision中数据集的使用

```python
import torchvision
from torch.utils.tensorboard import SummartWriter

dataset_transform = torchvision.transforms.Compose([
	torchvision.transforms.ToTensor()
])
train_set = torchvision.dataset.CIFAR10(root="/dataset", train=True, 
                                        transform=dataset_transform, download=True)
test_set = torchvision.dataset.CIFAR10(root="/dataset", train=False, 									  transform=dataset_transform,
download=True)

print(train_set[0])

writer = SummaryWriter("logs")
for i in range(10):
    img,target = test_set[i];
    writer.add_image("test_set", img, i)
    
writer.close()
```

![image-20210513210358430](https://raw.githubusercontent.com/youminglan/Picture/main/img/20210513210405.png)

