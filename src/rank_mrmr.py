# -*- coding: utf-8 -*-
import pymrmr
import numpy as np

def rankmrmr(ways,XtrainR,YtrainR):
    XtrainR.columns = XtrainR.columns.astype(str)
    res=pymrmr.mRMR(XtrainR, 'MIQ', XtrainR.shape[0])
    X_train=XtrainR.loc[:,res]
    print(X_train.columns)
    # np.savetxt('./out/inte_data/'+ways+'rank_colunms.txt',np.array(X_train.columns.tolist()),delimiter='\t',fmt='%s')    
    return X_train