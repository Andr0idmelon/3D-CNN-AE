from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset,DataLoader
import pickle
import torch

class MyDataset(Dataset):
    def __init__(self,data) :
        self.data=torch.from_numpy(data)
        self.data=self.data.float() 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.data[index],self.data[index])

class MyDataModule(LightningDataModule):
    def __init__(self,params):
        super().__init__()
        self.batchsize=params.data.batchsize

    #只在gpu0上调用一次，数据要在这里处理好
    def prepare_data(self):
        #分开涡量和压力 涡量
        with open(r'myModule/case1-128-128-128-wx-train.pickle','rb') as f:
            train_data=pickle.load(f)[:,0:1,:,:,:]
        train_set=MyDataset(train_data)
        del train_data
        with open(r'myModule/case1-128-128-128-wx-val.pickle','rb') as s:
            val_data=pickle.load(s)[:,0:1,:,:,:]
        val_set=MyDataset(val_data)
        del val_data
        with open('./dataset.pickle','wb') as p:
            pickle.dump((train_set,val_set),p,protocol = 4)

    #在每个gpu上都会调用,在这里指出哪些是训练集、验证集
    def setup(self,stage=None):
        with open('./dataset.pickle','rb') as g:
            self.train_data,self.val_data=pickle.load(g)
  
    def train_dataloader(self) :
        train_loader=DataLoader(dataset=self.train_data,batch_size=self.batchsize,shuffle=True,num_workers=5,pin_memory=True)
        return train_loader

    def val_dataloader(self) :
        val_loader=DataLoader(dataset=self.val_data,batch_size=self.batchsize,shuffle=False,num_workers=5,pin_memory=True)
        return val_loader