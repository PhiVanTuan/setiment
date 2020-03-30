import torch
import numpy as np

def squeeze_test():
    array=np.arange(start=0, stop=20, step=1).reshape((4,1,5))
    array=np.array(array)
    x=torch.from_numpy(array)
    print(array)
    y=torch.squeeze(x,1)
    print(y)

def un_squeeze_test():
    array=np.arange(start=0, stop=20, step=1).reshape((4,1,5))
    x = torch.tensor(array)
    y=torch.unsqueeze(x, 3)
    print(y)
def cat_test():
    array1=torch.tensor(np.arange(start=0, stop=6, step=1).reshape((1,2,3)))
    array2=torch.tensor(np.arange(start=6, stop=12, step=1).reshape((1,2,3)))
    list=[array1,array2]
    x=torch.cat(list,0)
    print(x)
cat_test()