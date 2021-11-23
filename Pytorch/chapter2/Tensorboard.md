### Tensorboard的使用

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "data/train/ants_image/00001.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(img_array.shape)

writer.add_image("test", img_array, 1)


#画函数
for i in range(100):
    writer.add_scalar("y=x", i, i)
    
writer.add_scalar()

```

按住ctrl键查看这个类的用法(源码)

![image-20210511180941575](https://raw.githubusercontent.com/youminglan/Picture/main/img/20210511180941.png)

查看tensorboard生成的文件：Terminal下

![image-20210511181934131](https://raw.githubusercontent.com/youminglan/Picture/main/img/20210511181934.png)