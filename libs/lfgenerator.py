import numpy as np
import pickle
import numpy as np
from math import exp, sqrt
import random   
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel





class AbstractGenerator(object):
    """
    Abstract generator class of input-output time series
    """

    def __init__(self, config={}):

        generator_config = self.get_default_config()
        generator_config.update(config)
        # The out_put length parameter, if not specified, set to same as input.
        if 'out_len' not in generator_config:
            generator_config['out_len'] = generator_config.get('path_len')

        self.build_generator(generator_config)
        self.config = generator_config

    def build_generator(self, config):
        """
        Build generator
        """
        raise NotImplementedError

    def get_default_config(self):
        """
        Returns default config
        """
        return {
            'input_dim': 1,
            'data_num': 128,
            'path_len': 32,
        }

    def generate_inputs(self, data_num, path_len):
        raise NotImplementedError

    def generate_outputs(self, inputs):
        raise NotImplementedError

    def generate(self,
                 data_num=None,
                 path_len=None):
        inputs = self.generate_inputs(data_num or self.config['data_num'],
                                      path_len or self.config['path_len'])
        outputs = self.generate_outputs(inputs)
        
        return inputs, outputs

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)


class LFGenGenerator(AbstractGenerator):
    """
    Generates a time series with general relation.
    For negative indexing we use following
    x =        [1, 2, 3, 4]
             -3   -2   -1   0
    index    0,-1,-2,-3
        y_{t+1} = H_{t}(x_{0:path_length})
                = int_{0}^{path_length} rho(t,s) x_{s} ds
        Inputs: x_{t}
        Outputs: y_{t}
    """

    def get_default_config(self):
        config = super().get_default_config()
        config.update({
            'input_dim': 1,
            'output_dim': 1,
            'dt': 0.1,
        })
        return config

    def build_generator(self, config):
        pass

    def rho(self, s, t):
        raise NotImplementedError

    def generate_inputs(self, batch_size, path_len):
        # random = 0.1 * np.random.normal(size=(
        #     batch_size,
        #     path_len,
        #     self.config['input_dim'],
        # ))
        random = np.random.rand(batch_size,
            path_len,
            self.config['input_dim'])
        return random

    def generate_outputs(self, inputs):
        _, path_len, _ = inputs.shape
        out_len = self.config['out_len']
        dt = self.config['dt']
        rho = np.array([[self.rho(t*dt,s*dt) for s in range(path_len)] for t in range(out_len)])

        return np.einsum('bsd,tsd->bt',inputs,rho)[...,None] # A simpler way of the following summation.

        # for t in range(0, out_len):
        #     output = 0
        #     for s in range(0, path_len):
        #         output += inputs[:, s, :] * self.rho(
        #             t, s)
        #     outputs.append(output)
        # return np.asarray(outputs).transpose(1, 0, 2)

    def plot_rho(self, d=0, causal=False):
        # return an array which can be used to plot rho
        # if input is multidimension, d specifies which dimension to plot.
        # If is causal then plot rho(t-s).

        out_len = self.config['out_len']
        dt = self.config['dt']
        rho = np.array([[self.rho(t*dt,s*dt) for s in range(out_len)] for t in range(out_len)])


        if causal:
            x = np.arange(0,dt*out_len, dt)
            rho = np.flip(rho[...,d][-1])
            return (x, rho)
        else:
            return rho[...,d]
        




class Shift(LFGenGenerator):
    def get_default_config(self):
        config = super().get_default_config()
        config.update({'dt':1, 'shift':[2]})
        
        return config

    def __repr__(self) -> str:
        return f"Shift_{self.config['shift']}"

    def rho(self, t, s):
        assert self.config['input_dim'] == len(self.config['shift']), "dimension not match"

        rho = []
        for k in self.config['shift']:
            if t-s == k:
                rho.append(1.0)
            else:
                rho.append(0.0)

        return rho

class Exponential(LFGenGenerator):
    def get_default_config(self):
        config = super().get_default_config()
        config.update({'dt':0.1, 'lambda':[0.5]})
        
        return config

    def rho(self, t, s):
        assert self.config['input_dim'] == len(self.config['lambda']), "dimension not match"

        if s <= t:
            return np.exp(-np.array(self.config['lambda']) * (t-s)/self.config['dt'])
        else:
            return np.zeros(self.config['input_dim'])



class TwoPart(LFGenGenerator):
    """
    Consists of two peak, centers describe center of peak, sigmas are size of each peak.
    """
    def get_default_config(self):
        config = super().get_default_config()
        config.update({'centers': [[1.5, 4]],
                       'sigmas':[[4, 4]],
                       'path_len':64})
        return config


    def rho(self, t,s):
        res = 0

        for center, sigma in zip(self.config['centers'], self.config['sigmas']):
            for cen, sig in zip(center, sigma):
                if s<=t:
                    res += exp(-((t-s)-cen)**2 * sig)

        return [res]
        

class ExpPeak(LFGenGenerator):

    def get_default_config(self):
        config = super().get_default_config()
        config.update({'lambda': [1],
                        'centers': [8],
                       'sigmas':[10],
                       'path_len':128})
        return config


    def rho(self, t,s):
        res = []

        if s <= t:
            for lam, center, sigma in zip(self.config['lambda'], self.config['centers'], self.config['sigmas']):
                res.append(np.exp(-np.array(lam) * (t-s)) + exp(-((t-s)-center)**2 * sigma))
        else:
            res.append(0)

        return res
    
class LorenzConstFGenerator(AbstractGenerator):
    """
    Generates Lorenz system

        xk_{t+1} = xk_{t} + dt * (xk-1_{t}*(xk+1_{t} -
                   xk-2_{t} - xk_{t} + F + zk_{t}), k=1, ..., K
        yj,k_{t+1} = yj,k_{t} + dt/eps * (
                     yj,k-1_{t}*(yj,k+1_{t} - yj,k-2_{t}
                     - yj,k_{t} + hy*xk_{t}),
                     j=1, ..., J
        zk_{t} = hx*sum_{j=1}^J{yj,k_{t}} / J

        Parameters
            * dt: time step

        Inputs: xk_{t}
        Ouputs: xk_{t+t_shift}
    """
    name = 'LorenzConstFGenerator'

    def generate_inputs(self,data_num, path_len):
        input = []
        for _ in range(data_num):
            data = self._generate_gaussian(path_len)
            input.append(data[:,np.newaxis])
        return np.array(input)

    def get_default_config(self):
        config = super().get_default_config()
        config.update({
            'dt': 0.01,
            'ndt': 20,
            'F': 4,
            'eps': 0.5,
            'hx': -1,
            'hy': 1,
        })
        return config

    def calibrate(self, path_len=128):
        inputs, outputs = self.generate(data_num=32,
                                        path_len=path_len,
                                        scale=False)
        # self.scale = np.linalg.norm(y) / 32
        self.out_scale = np.max(np.abs(inputs))
        self.in_scale = np.max(np.abs(outputs))
        K = self.config['K']
        self.config.update({'input_dim': K, 'output_dim': K})

    def build_generator(self, config):
        self.config = config
        #initial condition
        self.x =  _generate_gaussian(config['n_init'], self.config['K'],1).squeeze(-1)
        self.y = _generate_gaussian(self.config['n_init'], self.config['K'], self.config['J'])
        # burn in at beginning
        self.generate(data_num=self.config['n_init'],
                      path_len=100,
                      scale=False)
        pass

    def x_force(self):
        xkm1 = np.roll(self.x, 1, axis=-1)
        xkm2 = np.roll(self.x, 2, axis=-1)
        xkp1 = np.roll(self.x, -1, axis=-1)
        cyclic = xkm1 * (xkp1 - xkm2) - self.x
        dxdt = cyclic + np.mean(self.y, axis=-1) * self.config['hx']
        return dxdt

    def y_force(self):
        ykm1 = np.roll(self.y, 1, axis=-1)
        ykp1 = np.roll(self.y, -1, axis=-1)
        ykp2 = np.roll(self.y, -2, axis=-1)
        cyclic = ykp1 * (ykm1 - ykp2) - self.y
        dydt = (cyclic +
                self.x[:, :, None] * self.config['hy']) / self.config['eps']
        return dydt



    def generate(self,
                 data_num=None,
                 path_len=None,
                 scale=True,
                 train_data=True):
        data_num = data_num or self.config['data_num']
        path_len = path_len or self.config['path_len']
        pre_size = int(np.ceil(
            data_num / self.config['n_init'])) * self.config['n_init']
        inputs = np.zeros((pre_size, path_len, self.config['K']))
        outputs = np.zeros((pre_size, path_len, self.config['K']))
        x_path = np.zeros(
            (self.config['n_init'], path_len * 2, self.config['K']))
        batch_idx = 0
        while batch_idx < data_num - 1:
            x_path[:, 0] = self.x.copy()
            for t in range(2 * path_len - 1):
                for _ in range(self.config['ndt']):
                    dxdt = self.x_force() + self.config['F']
                    dydt = self.y_force()
                    self.x += dxdt * self.config['dt']
                    self.y += dydt * self.config['dt']
                x_path[:, t + 1] = self.x.copy()
            inputs[batch_idx:batch_idx +
                             self.config['n_init']] = x_path[:, :path_len].copy()
            outputs[batch_idx:batch_idx +
                              self.config['n_init']] = x_path[:, path_len:].copy()
            batch_idx += self.config['n_init']

        inputs = inputs[:data_num]
        outputs = outputs[:data_num]
        if scale:
            self.calibrate()
            assert hasattr(self, 'in_scale') and hasattr(
                self, 'out_scale'), 'call calibrate to obtain scale'
            outputs /= self.out_scale
            inputs /= self.in_scale
        if self.config['only_terminal']:
            outputs = outputs[:, -1, :]
        return inputs, outputs

def _generate_gaussian(num, seq_length, dim):
    xs = np.arange(seq_length)*0.1
    mean = [0 for _ in xs]
    gram = rbf_kernel(xs[:,np.newaxis], gamma=0.5)
        
    ys = np.random.multivariate_normal(mean, gram, size=(num,dim))
    return ys.transpose(0,2,1)

class LorenzRandFGenerator(LorenzConstFGenerator):
    """
    Generates Lorenz system

        xk_{t+1} = xk_{t} + dt * (xk-1_{t}*(xk+1_{t}
                   - xk-2_{t} - xk_{t} + Fk_{t} + zk_{t}), k=1, ..., K
        yj,k_{t+1} = yj,k_{t} + dt/eps * (yj,k-1_{t}*(yj,k+1_{t}
                     - yj,k-2_{t} - yj,k_{t} + hy*xk_{t}), j=1, ..., J
        zk_{t} = hx*sum_{j=1}^J{yj,k_{t}} / J

        Parameters
            * dt: time step

        Inputs: Fk_{t}
        Ouputs: xk_{t}
    """
    name = 'LorenzRandFGenerator'

    

    def get_default_config(self):
        config = super().get_default_config()
        config.update({
            'dt': 0.01,
            'ndt': 10,
            'f_low': 0.6,
            'f_high': 1.2,
            'eps': 0.5,
            'hx': -1,
            'hy': 1,
        })
        return config

    def generate(self,
                 data_num=None,
                 path_len=None,
                 scale=True,
                 train_data=True):
        data_num = data_num or self.config['data_num']
        path_len = path_len or self.config['path_len']
        pre_size = int(np.ceil(
            data_num / self.config['n_init'])) * self.config['n_init']
        inputs = np.zeros((pre_size, path_len, self.config['K']))
        outputs = np.zeros((pre_size, path_len, self.config['K']))
        outputs_sub = np.zeros(
            (self.config['n_init'], path_len, self.config['K']))
        batch_idx = 0
        while batch_idx < data_num - 1:
            # random force as input
            inputs_sub = _generate_gaussian(self.config['n_init'],path_len - 1,self.config['K'])                            
            # initial condition is necessary for prediction
            inputs_sub = np.concatenate([self.x[:, None, :], inputs_sub],
                                        axis=1)
            outputs_sub[:, 0] = self.x.copy()
            for t in range(path_len - 1):
                for _ in range(self.config['ndt']):
                    dxdt = self.x_force() + inputs_sub[:, t + 1]
                    dydt = self.y_force()
                    self.x += dxdt * self.config['dt']
                    self.y += dydt * self.config['dt']
                outputs_sub[:, t + 1] = self.x.copy()
            inputs[batch_idx:batch_idx +
                             self.config['n_init']] = inputs_sub.copy()
            outputs[batch_idx:batch_idx +
                              self.config['n_init']] = outputs_sub.copy()
            batch_idx += self.config['n_init']

        inputs = inputs[:data_num]
        outputs = outputs[:data_num]
        if scale:
            self.calibrate()
            assert hasattr(self, 'in_scale') and hasattr(
                self, 'out_scale'), 'call calibrate to obtain scale'
            outputs /= self.out_scale
            inputs = (inputs - self.config['f_low']) / (self.config['f_high'] -
                                                        self.config['f_low'])
        if self.config['only_terminal']:
            outputs = outputs[:, -1, :]
        return inputs, outputs