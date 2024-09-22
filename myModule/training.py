
from myModule.runcontrol import autoencoder
from pytorch_lightning.loggers import WandbLogger
from myModule.datawork import MyDataModule
from myModule.callbacks import set_callbacks
from pytorch_lightning import Trainer



def training(params):
    wandb_logger=WandbLogger(name='demo_name',project="project_name",log_model=True)
    data=MyDataModule(params)
    model=autoencoder(params)
    callbacks=set_callbacks(params)
    trainer=Trainer(max_epochs=params.training.training_epoch,callbacks=callbacks,logger=wandb_logger,precision=16,accelerator='gpu',devices=[0]
                    ,strategy="ddp_find_unused_parameters_false")
    trainer.fit(model,data)




    

