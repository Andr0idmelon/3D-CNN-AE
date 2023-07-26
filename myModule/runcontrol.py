import pytorch_lightning as pl
from myModule.model import network
from torch import optim
import torchmetrics

class autoencoder(pl.LightningModule):

    def __init__(self,params):
        super().__init__()
        self.lr=params.training.lr
        self.model=network(params)
        self.save_hyperparameters(params)
        self.loss_fn=torchmetrics.MeanSquaredError()


    def forward(self,x):
        return self.model(x)


    def configure_optimizers(self):
        optimizer=optim.Adam(self.parameters(),lr=self.lr)
        milestones=[8000,12000,15000,18000]
        scheduler = {'scheduler': optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1),'interval': 'epoch' }
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}#
        return optim_dict

    def training_step(self,batch,batch_idx):
        x,y=batch
        pred=self.model(x)
        loss=self.loss_fn(pred,y)
        self.log('train_loss',loss)
        return loss
    

    def validation_step(self,batch,batch_index):
        x,y=batch
        pred=self.model(x)
        loss=self.loss_fn(pred,y)
        self.log('val_loss',loss)

