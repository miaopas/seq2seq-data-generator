Please clone the repository: https://github.com/petrobras/3W.git
Original dataset can be find under dataset directory, folder 0 contains nomal condition, while folder 4 represent flow instability fault. Please put the script 3w_generator.py under 3W-main, outside dataset folder.

During a flow instability, at least one of the monitored variables undergoes relevant changes but with tolerable amplitudes. If can predict and take action in advance,  can avoids all the negative aspects associated with this more severe anomaly.
Typical window of this kind of fault is 15mins (900 steps, as the sample rate is 1HZ). All normal data are labelled with 0, all fault data are labelled with 1.

Run 3w_generator.py will generate train and test datasets with the shape (batch, length, n_dim = 3), all 3 variables are included in n_dim:
1. Pressure at temperature and pressure transducer (TPT); 
2. Temperature at the TPT; 
3. Pressure upstream of the Production Choke (PCK).

The labels are 1d array.

Without overlap and with a 900 timestep window, we have 9002 instances for train and 3909 for test.

