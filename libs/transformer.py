import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Linear
from torch.nn.modules.utils import _pair
import math



class Embeddings(nn.Module):
    """Construct the embeddings.
    """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.embeddings = torch.nn.Linear(in_features=config['input_dim'], out_features=config['hidden_size'])
        self.position_embeddings = None
        if config['position_embeddings']:
            self.position_embeddings = nn.Parameter(torch.zeros(1, config['input_length'], config["hidden_size"]))


    def forward(self, x):
        embeddings = self.embeddings(x)
        

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        return embeddings


class TransformerModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.embedding = Embeddings(config)
        transformer_layer = nn.TransformerEncoderLayer(config['hidden_size'], nhead=config['nhead'], batch_first=True)
        self.transofmer = nn.TransformerEncoder(transformer_layer, config['num_layers'])

        self.out = nn.Linear(config['hidden_size'], config["output_dim"])
        self.config = config
        self.save_hyperparameters()

    def forward(self, x):
        x = self.embedding(x)
        
        y = self.transofmer(x)

        y = self.out(y)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['lr'])
        scheduler = {
                        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5), 
                        "interval": "epoch",
                        "frequency": 1,
                        "monitor": "train_loss"
                    }
        return {"optimizer": optimizer, "lr_scheduler":scheduler}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        train_loss = self.config['loss'](y_hat, y)

        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        
        self.log("lr", lr, prog_bar=True, on_step=True, sync_dist=True)


        self.log("train_loss", train_loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        return train_loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        valid_loss = self.config['loss'](y_hat, y)

        self.log("valid_loss", valid_loss, prog_bar=True, logger=True, sync_dist=True)
        return valid_loss

    def validation_epoch_end(self, outputs):
        pass

    def predict(self, x):
        pass


5     