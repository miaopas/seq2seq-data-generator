from libs.train import train_model
from libs.seq2seq_model import RNNModel, TCNModel, S4DModel
from libs.transformer import TransformerModel
from torch import nn
import torch
import numpy as np

def get_slices(array, slice_length, overlap):
    num_slices = (len(array) - overlap) // (slice_length - overlap)

    # Initialize an empty list to store the slices
    slices = []

    # Create overlapping slices
    for i in range(num_slices):
        start = i * (slice_length - overlap)
        end = start + slice_length
        slice_data = array[start:end]
        slices.append(slice_data)
    return np.array(slices)

def train_cfd_rnn(re, index1, index2):

        
        config = {'input_dim': 2, 'output_dim':2, 'hid_dim':128, 'num_layers':2, 'loss':nn.MSELoss(), 'optim':'Adam', 'lr':0.001}

        model =RNNModel(config)
        

        # model = RNNModel.load_from_checkpoint("runs/cfd_0_4/version_3/checkpoints/cfd_0_4-epoch=399-valid_loss=8.89e-04.ckpt")
        data = []
        for i in range(6):
            data.append(np.load(f"cyl_cfd/Re{re}/p{i+1}.npy"))


        x = np.array(np.array_split(data[index1], 400)[1:])[:,:,[1,2]]
        y = np.array(np.array_split(data[index2], 400)[1:])[:,:,[1,2]]

    
        

        train_model(f"cfd_Re{re}_{index1}_{index2}_RNN",model,x,y,0.75,batch_size=40,epochs=30,devices=1)


def train_cfd_tcn(re, index1, index2):

        
        config = {
        "input_dim": 2,
        "output_dim": 2,
        "loss": nn.MSELoss(),  # This should be fixed
        "optim": "Adam",
        "lr": 1e-3,
        "kernel_size": 2,
        "channels": 64,
        "layers": 10,  # This is a parameter list for the model
    }

        model =TCNModel(config)
        
        data = []
        for i in range(6):
            data.append(np.load(f"cyl_cfd/Re{re}/p{i+1}.npy"))

        # index1, index2 = 0, 1

        x = np.array(np.array_split(data[index1], 400)[1:])[:,:,[1,2]]
        y = np.array(np.array_split(data[index2], 400)[1:])[:,:,[1,2]]

        train_model(f"cfd_Re{re}_{index1}_{index2}_TCN",model,x,y,0.75,batch_size=40,epochs=30,devices=1)



def train_cfd_rnn1(re, index1, index2):

        
        config = {'input_dim': 2, 'output_dim':2, 'hid_dim':128, 'num_layers':2, 'loss':nn.MSELoss(), 'optim':'Adam', 'lr':0.001}

        model =RNNModel(config)
        

        # model = RNNModel.load_from_checkpoint("runs/cfd_0_4/version_3/checkpoints/cfd_0_4-epoch=399-valid_loss=8.89e-04.ckpt")
        data = []
        for i in range(6):
            data.append(np.load(f"cyl_cfd/Re{re}/p{i+1}.npy"))



        data = []

        for i in range(6):
            data.append(np.load(f"cyl_cfd/Re{re}/p{i+1}.npy"))

        x = get_slices(data[index1], 64, 32)[...,[1,2]]
        y = get_slices(data[index2], 64, 32)[...,[1,2]]


    
        

        train_model(f"cfd_Re{re}_{index1}_{index2}_RNN",model,x,y,0.75,batch_size=40,epochs=30,devices=1)


def train_cfd_tcn1(re, index1, index2):

        
        config = {
        "input_dim": 2,
        "output_dim": 2,
        "loss": nn.MSELoss(),  # This should be fixed
        "optim": "Adam",
        "lr": 1e-3,
        "kernel_size": 2,
        "channels": 64,
        "layers": 10,  # This is a parameter list for the model
    }

        model =TCNModel(config)
        
        
        data = []

        for i in range(6):
            data.append(np.load(f"cyl_cfd/Re{re}/p{i+1}.npy"))

        x = get_slices(data[index1], 64, 32)[...,[1,2]]
        y = get_slices(data[index2], 64, 32)[...,[1,2]]


        train_model(f"cfd_Re{re}_{index1}_{index2}_TCN",model,x,y,0.75,batch_size=40,epochs=30,devices=1)



def train_cfd_transformer(re, index1, index2):

    
    data = []
    for i in range(6):
        data.append(np.load(f"cyl_cfd/Re{re}/p{i+1}.npy"))

    x = get_slices(data[index1], 64, 32)[...,[1,2]]
    y = get_slices(data[index2], 64, 32)[...,[1,2]]




    config = {'input_length':x.shape[1],'input_dim':2,  'output_dim':2, 'position_embeddings':True,
                 'hidden_size':32, 'nhead': 4, 'num_layers':4,
                  'lr':1e-2, 'gradient_accumulation_steps':1, 'loss':torch.nn.MSELoss() }
    
    model = TransformerModel(config)



    train_model(f"0000cfd_Re{re}_{index1}_{index2}_Transformer",model,x,y,0.75,batch_size=40,epochs=30,devices=1)


def train_cfd_s4d(re, index1, index2):

    
    data = []
    for i in range(6):
        data.append(np.load(f"cyl_cfd/Re{re}/p{i+1}.npy"))

    x = get_slices(data[index1], 64, 32)[...,[1,2]]
    y = get_slices(data[index2], 64, 32)[...,[1,2]]



    config = {'lr':1e-2, 'gradient_accumulation_steps':1, 'loss':torch.nn.MSELoss(), 'optim':'Adam' }
    
    model = S4DModel(hid_dim=128, input_dim=2, output_dim=2, config=config)



    train_model(f"1cfd_Re{re}_{index1}_{index2}_S4D",model,x,y,0.75,batch_size=512,epochs=30,devices=1)
