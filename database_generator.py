# -*- coding: utf-8 -*-
from batch_gen_util import shift_gen, exp_gen, twopart_gen, exppeak_gen




if __name__ == '__main__':
    num_per_para = 100000  # number of series for each combination of parameters
    shift_gen(num_per_para)
    exp_gen(num_per_para)
    twopart_gen(num_per_para)
    exppeak_gen(num_per_para)

