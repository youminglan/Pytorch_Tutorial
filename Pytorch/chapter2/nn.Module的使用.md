### nn.Module的使用

```python
import torch
from torch import nn

class Model(nn.Moduel):
    def __init__(self):
        super.().__init__()
        
    def forward(self, input):
        output = input + 1
        
model = Model()
x = torch.tensor(1.0)
output = model(x)
print(output)
```

