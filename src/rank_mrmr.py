# -*- coding: utf-8 -*-
import pymrmr
import numpy as np
import pandas as pd
def rankmrmr(ways,XtrainR,YtrainR):
    data=pd.concat([YtrainR,XtrainR],axis=1)
    data.columns = data.columns.astype(str)
    XtrainR.columns=XtrainR.columns.astype(str)
    res=pymrmr.mRMR(data, 'MIQ', XtrainR.shape[1])
    X_train=XtrainR.loc[:,res]
    print(X_train.columns)
    # np.savetxt('./out/inte_data/'+ways+'rank_colunms.txt',np.array(X_train.columns.tolist()),delimiter='\t',fmt='%s')    
    return X_train
