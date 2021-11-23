### Dataset

提供一种方式去获取数据及其label

1. 如何获取每一个数据及其label

2. 告诉我们总共有多少个数据

   ```python
   from torch.utils.data import Dataset
   import os
   from PTL import Image
   class MyData(Dataset):
       
       def __init__(self, root_dir, label_dir):
           self.root_dir = root_dir
           self.label_dir = label_dir
           self.path = os.path.join(self.root_dir, self.label_dir)
           self.img_path = os.listdir(self.path)
           
       def __getitem__(self,dix):
           img_name = self.img_path[idx]
           img_item_path = os.path.join(self.root_dir, self.label_dir)
           img = Image.open(img_item_path)
           label = self.label_dir
           return img, label
       
       def __len__(self):
           return len(self.img_path)
       
   root_dir = "dataset/train"    
   ants_label_dir = "ants"
   bees_label_dir = "bees"
   ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)   
   
   train_dataset = ants_dataset + bees_dataset
   ```
   
   

### Dataloader

为神经网络提供不同的数据形式