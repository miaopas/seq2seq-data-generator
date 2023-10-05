from libs.train_cfd import *
# from libs.train_with_tune import *


for index1 in [0,1,2,3,4]:
    for index2 in [1,2,3,4,5]:
        for re in [60, 100, 250]:
            if index1 != index2:
                print(f'Re{re}_{index1}_{index2}')
                train_cfd_transformer(re, index1, index2)


