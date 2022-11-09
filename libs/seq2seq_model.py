import pytorch_lightning as pl
import torch
from torch import nn
from libs.layers import LinearRNN


class Seq2SeqModel(pl.LightningModule):
    def __init__(self, loss=nn.MSELoss(), optim='Adam'):
        super().__init__() 
        self.loss = loss #############
        self.optim = optim
        self.save_hyperparameters(ignore=['loss'])

    def forward(self, x):
        pass

    def configure_optimizers(self):
        if self.optim == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        elif self.optim == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        else:
            optimizer = torch.optim.SparseAdam(self.parameters(), lr=1e-3)

        scheduler = {
                        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer), 
                        "interval": "epoch",
                        "frequency": 1,
                        "monitor": "train_loss_epoch"
                    }
        # return {"optimizer": optimizer}

        return {"optimizer": optimizer, "lr_scheduler":scheduler}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        trainloss = self.loss(y_hat, y)
        self.log("train_loss", trainloss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return trainloss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        validloss = self.loss(y_hat, y)
        self.log("valid_loss", validloss, prog_bar=True, logger=True, sync_dist=True)
        return validloss

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            pred = self(x)
            return pred.detach().cpu().numpy()


class RNNModel(Seq2SeqModel):
    def __init__(self, hid_dim, num_layers, input_dim, output_dim):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hid_dim, num_layers=num_layers, batch_first=True, )
        self.dense = nn.Linear(hid_dim, output_dim)

    def forward(self, x):
        y = self.rnn(x)[0]
        y = self.dense(y)
        output = y
        return output

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

class LinearRNNModel(Seq2SeqModel):
    def __init__(self, hid_dim, input_dim, output_dim):
        super().__init__()
        self.rnn = LinearRNN(input_dim=input_dim, output_dim=output_dim, hid_dim=hid_dim)


    def forward(self, x):
        y = self.rnn(x)

        return y