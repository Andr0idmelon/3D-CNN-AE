from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
def set_callbacks(params):
    if params.training.early_stop==True:
        early_stopping=EarlyStopping(
            monitor=params.training.early_stop_settings.monitor,
            patience=params.training.early_stop_settings.patience,
            verbose=params.training.early_stop_settings.verbose)
    checkpoint=ModelCheckpoint(monitor='val_loss',save_last=True,filename='{epoch}_{val_loss:.3f}',mode='min')
    callbacks=[checkpoint,early_stopping]
    return callbacks