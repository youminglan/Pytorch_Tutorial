### Data_loader的使用

```python
from torchvision
from torch.utils.data import dataLoader
test_set = torchvision.dataset.CIFAR10(root="/dataset", train=False								   		transform=torchvision.transforms.ToTensor(),download=True)

test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_worker=0, drop_last=False)

#数据集中的第一张图片
img, target = test_data[0]
pring(img.shape)
print(target)

#一个batch
writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data;
        #print(imgs.shape)
        #print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1
 
writer.close
```

![image-20210513212040156](https://raw.githubusercontent.com/youminglan/Picture/main/img/20210513212040.png)

![image-20210513213306339](https://raw.githubusercontent.com/youminglan/Picture/main/img/20210513213306.png)