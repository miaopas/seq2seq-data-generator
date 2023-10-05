import pytorch_lightning as pl
import torch
from torch import nn
from libs.layers import LinearRNN, ComplexLinearRNN, LinearTCN
from libs.tcn import TemporalConvNet

from libs.s4d import S4D
from libs.s4 import S4


class Seq2SeqModel(pl.LightningModule):
    # def __init__(self, loss=nn.MSELoss(), optim='Adam'):
    def __init__(self, config):
        super().__init__()
        self.loss = config["loss"]  #############
        self.optim = config["optim"]
        self.lr = config["lr"]
        self.save_hyperparameters(ignore=["loss"])

    def forward(self, x):
        pass

    def configure_optimizers(self):
        if self.optim == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            "interval": "epoch",
            "frequency": 10,
            "monitor": "train_loss_epoch",
        }
        # return {"optimizer": optimizer}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        trainloss = self.loss(y_hat, y)
        self.log(
            "train_loss",
            trainloss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
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
    def __init__(
        self,
        config,
        return_sequence=True,
        dtype=64,
    ):
        super().__init__(config)
        self.rnn = nn.RNN(
            input_size=config["input_dim"],
            hidden_size=config["hid_dim"],
            num_layers=config["num_layers"],
            batch_first=True,
        )
        self.dense = nn.Linear(
            config["hid_dim"],
            config["output_dim"],
        )

        self.return_sequence = return_sequence

    def forward(self, x):
        y = self.rnn(x)[0]
        y = self.dense(y)
        output = y

        if self.return_sequence:
            return output
        else:
            return output[:, -1, :]  # return last output

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0)


class LinearRNNModel(Seq2SeqModel):
    # def __init__(self, hid_dim, input_dim, output_dim):
    def __init__(self, config):
        super().__init__(config)
        self.rnn = LinearRNN(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            hid_dim=config["hid_dim"],
        )

    def forward(self, x):
        y = self.rnn(x)

        return y


class ComplexLinearRNNModel(Seq2SeqModel):
    def __init__(self, hid_dim, input_dim, output_dim):
        super().__init__()
        self.rnn = ComplexLinearRNN(
            input_dim=input_dim, output_dim=output_dim, hid_dim=hid_dim
        )

    def forward(self, x):
        y = self.rnn(x)

        return y


class TCNModel(Seq2SeqModel):
    # def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
    def __init__(self, config):
        super(TCNModel, self).__init__(config)

        channel_list = [config["channels"] for _ in range(config["layers"])]
        self.tcn = TemporalConvNet(
            config["input_dim"], channel_list, kernel_size=config["kernel_size"]
        )
        self.linear = nn.Linear(channel_list[-1], config["output_dim"])
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)
        y1 = self.tcn(x)
        y1 = y1.permute(0, 2, 1)
        return self.linear(y1)


class LinearTCNModel(Seq2SeqModel):
    def __init__(self, config):
        super().__init__(config)
        self.tcn = LinearTCN(
            config["input_dim"],
            config["output_dim"],
            config["channels"],
            config["layers"],
        )

    def forward(self, x):
        y = self.tcn(x)

        return y


class S4DModel(Seq2SeqModel):
    def __init__(self, hid_dim, input_dim, output_dim, config):
        super().__init__(config)
        self.rnn = S4D(hid_dim)
        
        self.input = nn.Linear(input_dim, hid_dim)
        self.output = nn.Linear(hid_dim, output_dim)

    def forward(self, x):
        # Input (B,L,H)

        x = self.input(x)

        x = x.permute(0,2,1)
        # (B,H,L)
        y = self.rnn(x)[0]
        y = y.permute(0,2,1)

        y = self.output(y)

        return y

# class S4Model(Seq2SeqModel):
#     def __init__(self, hid_dim, output_dim):
#         super().__init__()
#         self.rnn = S4(hid_dim)
#         self.output = nn.Linear(hid_dim, output_dim)

#     def forward(self, x):
#         # Input (B,L,H)

#         x = x.permute(0,2,1)
#         # (B,H,L)
#         y = self.rnn(x)[0]
#         y = y.permute(0,2,1)

#         y = self.output(y)

#         return y
