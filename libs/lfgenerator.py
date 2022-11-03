import numpy as np
import pickle
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from math import exp, sqrt
import random   
def _generate_gaussian(num, seq_length, dim):
    xs = np.arange(seq_length)*0.1
    mean = [0 for _ in xs]
    gram = rbf_kernel(xs[:,np.newaxis], gamma=0.5)
        
    ys = np.random.multivariate_normal(mean, gram, size=(num,dim))
    return ys.transpose(0,2,1)



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
            'output_dim': 1,
            'data_num': 128,
            'path_len': 32,
            'only_terminal': False
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

        return np.einsum('bsd,ts->btd',inputs,rho) # A simpler way of the following summation.

        # for t in range(0, out_len):
        #     output = 0
        #     for s in range(0, path_len):
        #         output += inputs[:, s, :] * self.rho(
        #             t, s)
        #     outputs.append(output)
        # return np.asarray(outputs).transpose(1, 0, 2)


class Shift(LFGenGenerator):
    def get_default_config(self):
        config = super().get_default_config()
        config.update({'shift':2})
        return config

    def rho(self, t, s):
        if t-s == self.config['shift']:
            return 1.0
        else:
            return 0.0


class LinearRNNGroundTruth(LFGenGenerator):
    """
    The ground truth for a linear RNN. 
    """
    name = 'LinearRNNGroundTruth'

    def get_default_config(self):
        config = super().get_default_config()
        return config

    def rho(self, t,s):

        if s <=t:
            return 0.5**(t-s)+ 0.6**(t-s)
        else:
            return 0

        


class TwoPart(LFGenGenerator):
    """
    The ground truth for a linear RNN. 
    """
    def get_default_config(self):
        config = super().get_default_config()
        config.update({'centers': [6, 25],
                       'sigmas':[0.5, 0.5]})
        return config


    def rho(self, t,s):
        res = 0
        for center, sigma in zip(self.config['centers'], self.config['sigmas']):
            res += exp(-(s-center)**2 * sigma)/4
        res += 0.5

        return res
        
