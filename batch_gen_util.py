# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle

from libs.lfgenerator import TwoPart, Shift, Exponential, ExpPeak



# data saving directory
savedir = 'Data'

# shift
def shift_gen(num_per_para):
    # makedir
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    shiftdir = os.path.join(savedir,'Shift')
    if not os.path.exists(shiftdir):
        os.mkdir(shiftdir)
        
    # parameters    
    para = {
        'input_dim' : 1,
        'path_len' : 256,
        'shift' : 1,
        'max_shift' : 100,
        'step' : 10,
        'num_per_para' : num_per_para
    }
    shifts = np.arange(0,para['max_shift']+1,para['step'])
   
    # generate
    for shift in shifts:
        para['shift'] = shift
        generator = Shift({'input_dim':para['input_dim'], 'path_len':para['path_len'] ,'shift':[shift]})
        input,output = generator.generate(data_num = para['num_per_para'])
        in_name = 'in_shift'+str(int(shift))+'.pickle'
        out_name = 'out_shift'+str(int(shift))+'.pickle'
        in_path = os.path.join(shiftdir,in_name)
        out_path = os.path.join(shiftdir,out_name)
        with open(in_path, 'wb') as file:
            pickle.dump(input,file)
        with open(out_path, 'wb') as file:
            pickle.dump(output,file)

        # calculate memory
        rho = generator.plot_rho(d=0, causal=True)
        memory = memory_calculater(rho[0],rho[1])
        para['memory'] = memory

        # write data summary
        save_summary(para, shiftdir, 'shift'+str(int(shift)))



def exp_gen(num_per_para):
    # makedir
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    expdir = os.path.join(savedir,'Exp')
    if not os.path.exists(expdir):
        os.mkdir(expdir)
        
    # parameters    
    para = {
    'input_dim' : 1,
    'path_len' : 256,
    'lambda' : 1,
    'num_per_para' : num_per_para
    }
    lambdas = np.linspace(0.1,1,num = 10)
    
    # generate
    for l in lambdas:
        para['lambda'] = l
        generator = Exponential({'input_dim':para['input_dim'], 'path_len':para['path_len'], 'lambda':[l]})   
        input,output = generator.generate(data_num = para['num_per_para'])
        in_name = 'in_exp'+'%.4f'%l+'.pickle'
        out_name = 'out_exp'+'%.4f'%l+'.pickle'
        in_path = os.path.join(expdir,in_name)
        out_path = os.path.join(expdir,out_name)
        with open(in_path, 'wb') as file:
            pickle.dump(input,file)
        with open(out_path, 'wb') as file:
            pickle.dump(output,file)
        
        # calculate memory
        rho = generator.plot_rho(d=0, causal=True)
        memory = memory_calculater(rho[0],rho[1])
        para['memory'] = memory

        # save data summary
        save_summary(para, expdir, 'exp'+'%.4f'%l)




def twopart_gen(num_per_para):
    # makedir
    if not os.path.exists(savedir):
        os.mkdir(savedir)    
    tpdir = os.path.join(savedir,'TwoPart')
    if not os.path.exists(tpdir):
        os.mkdir(tpdir)
        
    # parameters 
    d2s=np.linspace(5,10,11)
    d1s = ['close','far']
    sigma2s = [0.25,1,4]
    
    # generate
    for d1 in d1s:
        if d1 == 'close':
            for sigma2 in sigma2s: 
                for d2 in d2s:
                    name = str
                    para = {
                            'centers': [[d2/3, d2]],
                            'sigmas':[[1, sigma2]],
                            'path_len':256,
                            'num_per_para':num_per_para
                            }
                    generator = TwoPart({'centers': para['centers'],'sigmas':para['sigmas'],'path_len':para['path_len']})
                    input,output = generator.generate(data_num = para['num_per_para'])
                    in_name = 'in_TP_'+d1+'d'+'%.2f'%d2+'sigma'+'%.2f'%sigma2 +'.pickle'
                    out_name = 'out_TP_'+d1+'d'+'%.2f'%d2+'sigma'+'%.2f'%sigma2 +'.pickle'
                    in_path = os.path.join(tpdir,in_name)
                    out_path = os.path.join(tpdir,out_name)
                    with open(in_path, 'wb') as file:
                        pickle.dump(input,file)
                    with open(out_path, 'wb') as file:
                        pickle.dump(output,file)

                    # calculate memory
                    rho = generator.plot_rho(d=0, causal=True)
                    memory = memory_calculater(rho[0],rho[1])
                    para['memory'] = memory

                    # save data summary
                    save_summary(para,tpdir, 'TP_'+d1+'d'+'%.2f'%d2+'sigma'+'%.2f'%sigma2)

        if d1 == 'far':
            for sigma2 in sigma2s: 
                for d2 in d2s:
                    name = str
                    para = {
                            'centers': [[d2/3*2, d2]],
                            'sigmas':[[1, sigma2]],
                            'path_len':256,
                            'num_per_para':num_per_para
                            }
                    generator = TwoPart({'centers': para['centers'],'sigmas':para['sigmas'],'path_len':para['path_len']})
                    input,output = generator.generate(data_num = para['num_per_para'])
                    in_name = 'in_TP_'+d1+'d'+'%.2f'%d2+'sigma'+'%.2f'%sigma2 +'.pickle'
                    out_name = 'out_TP_'+d1+'d'+'%.2f'%d2+'sigma'+'%.2f'%sigma2 +'.pickle'
                    in_path = os.path.join(tpdir,in_name)
                    out_path = os.path.join(tpdir,out_name)
                    with open(in_path, 'wb') as file:
                        pickle.dump(input,file)
                    with open(out_path, 'wb') as file:
                        pickle.dump(output,file)

                    # calculate memory
                    rho = generator.plot_rho(d=0, causal=True)
                    memory = memory_calculater(rho[0],rho[1])
                    para['memory'] = memory

                    # save data summary
                    save_summary(para, tpdir, 'TP_'+d1+'d'+'%.2f'%d2+'sigma'+'%.2f'%sigma2)


def exppeak_gen(num_per_para):
    # makedir
    if not os.path.exists(savedir):
        os.mkdir(savedir)  
    epkdir = os.path.join(savedir,'ExpPeak')
    if not os.path.exists(epkdir):
        os.mkdir(epkdir)    
        
    # parameters
    centers = np.linspace(5,10,6)
    lambdas = [0.5,1,2]
    path_len = 256
    data_num = num_per_para
    
    # generate
    for center in centers:
        for l in lambdas:
            para = {
                'input_dim' : 1,
                'center' : [center],
                'sigmas' : [1],
                'path_len' : 256,
                'lambda' : [l],
                'num_per_para' : num_per_para
                }
            generator = ExpPeak({'lambda':[l],'centers': [center], 'sigmas':[1], 'path_len':path_len})
            input,output = generator.generate(data_num = data_num)
            in_name = 'in_epk_lambda'+'%.2f'%l+'d'+str(int(center))+'.pickle'
            out_name = 'out_epk_lambda'+'%.2f'%l+'d'+str(int(center))+'.pickle'
            in_path = os.path.join(epkdir,in_name)
            out_path = os.path.join(epkdir,out_name)
            with open(in_path, 'wb') as file:
                pickle.dump(input,file)
            with open(out_path, 'wb') as file:
                pickle.dump(output,file)
            
            # calculate memory
            rho = generator.plot_rho(d=0, causal=True)
            memory = memory_calculater(rho[0],rho[1])
            para['memory'] = memory

            # save data summary
            save_summary(para, epkdir, 'epk_lambda'+'%.2f'%l+'d'+str(int(center)))


def memory_calculater(x, rho):
    len = rho.shape[0]
    memory = np.zeros(len)
    
    for idx, i_rho in enumerate(rho):
        if idx < rho.shape[0] -1:
            memory[idx] = abs(rho[idx+1] - rho[idx]) * x[idx]
        else:
            memory[idx] = abs(rho[idx-1] - rho[idx]) * x[idx]

    memory = max(memory)
    return memory


def save_summary(para, dir, filename):
    # input the parameter list. write to file with given filename
    file = os.path.join(dir, filename+'.txt')
    with open(file,'w') as f:
        f.write('Data summary\n')
        f.write('-------------------------------------------\n')
        for key, value in para.items():
            f.write(str(key))
            f.write(': ')
            if isinstance(value, list):
                f.write(str(value[0]))
                f.write('\n')
            else:
                f.write(str(value))
                f.write('\n')


