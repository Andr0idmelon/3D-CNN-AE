import torch.nn as nn
import numpy as np


class network(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.batchsize=params.data.batchsize
        self.p=params.model.predict_windowslength
        num_eig=params.model.num_eig
        conv_settings=params.model.conv_settings
        maxpool_settings=params.model.maxpool_settings
        uppool_scale=params.model.uppool_settings
        self.activation=params.model.activation

        self.encoder=nn.Sequential(
            self.create_enclayer(1,4,[[5,5,5],[1,1,1],[2,2,2]],maxpool_settings,self.activation),
            self.create_enclayer(4,8,[[5,5,5],[1,1,1],[2,2,2]],maxpool_settings,self.activation),
            self.create_enclayer(8,16,[[5,5,5],[1,1,1],[2,2,2]],maxpool_settings,self.activation),
            self.create_enclayer(16,32,[[5,5,5],[1,1,1],[2,2,2]],maxpool_settings,self.activation)
        )

        self.fc_enc=nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(8*8*8*32,num_eig),
            nn.Unflatten(dim=1,unflattened_size=(1,num_eig))
        )


        self.fc_dec=nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(num_eig,8*8*8*32),
            nn.Unflatten(dim=1,unflattened_size=(32,8,8,8)),
        )

        self.decoder=nn.Sequential(
            self.create_declayer(32,16,[[5,5,5],[1,1,1],[2,2,2]],uppool_scale,self.activation),
            self.create_declayer(16,8,[[5,5,5],[1,1,1],[2,2,2]],uppool_scale,self.activation),
            self.create_declayer(8,4,[[5,5,5],[1,1,1],[2,2,2]],uppool_scale,self.activation),
            self.create_declayer(4,1,[[5,5,5],[1,1,1],[2,2,2]],uppool_scale,'tanh')
        )


    def make_eig(self,x):
        x=self.encoder(x)
        eig=self.fc_enc(x)
        return eig

    def forward(self,x):
        x=self.encoder(x)
        x=self.fc_enc(x)
        x=self.fc_dec(x)
        x=self.decoder(x)
        return x

    @staticmethod
    def create_enclayer(in_channel,out_channel,conv_settings,maxpool_settings,activation):
        acts=nn.ModuleDict([['tanh',nn.Tanh()],['relu',nn.ReLU()],['softsign',nn.Softsign()],['sigmoid',nn.Sigmoid()]])
        conv_kernelsize,conv_stride,conv_padding=conv_settings
        return nn.Sequential(
            nn.Conv3d(in_channel,out_channel,conv_kernelsize,conv_stride,conv_padding),
            nn.BatchNorm3d(num_features=out_channel),
            acts[activation],
            nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        )

    @staticmethod
    def create_declayer(in_channel,out_channel,conv_settings,uppool_scale,activation):
        acts=nn.ModuleDict([['tanh',nn.Tanh()],['relu',nn.ReLU()],['softsign',nn.Softsign()],['sigmoid',nn.Sigmoid()]])
        conv_kernelsize,conv_stride,conv_padding=conv_settings
        return nn.Sequential(
            nn.Upsample(scale_factor=uppool_scale),
            nn.Conv3d(in_channel,out_channel,conv_kernelsize,conv_stride,conv_padding),
            nn.BatchNorm3d(num_features=out_channel),
            acts[activation]
        )  
        

