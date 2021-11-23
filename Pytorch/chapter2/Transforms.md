### Transforms

![image-20210512122650292](https://raw.githubusercontent.com/youminglan/Picture/main/img/20210512122650.png)

python用法

```python
from torchvision import transforms
from  PIL import Image

# transform的使用

img_path = "dataset/train/anst/1111.jpg"
img = Image.open(img_path)
print(img)
tensor_trans = transforms.ToTersor()
tensor_img = tensor_trans(img)


# 为什么需要tensor数据类型

```

![image-20210512140238878](https://raw.githubusercontent.com/youminglan/Picture/main/img/20210512140238.png)

