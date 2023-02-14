import numpy as np
import pickle
import numpy as np
from math import exp, sqrt
import random   
import matplotlib.pyplot as plt





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
        random = 0.1 * np.random.normal(size=(
            batch_size,
            path_len,
            self.config['input_dim'],
        ))
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