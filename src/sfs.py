# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
import numpy as np
import prob
from arg import *
from loadfile import *


def SFS(X_train,YtrainR,outpath,outname):
    train_val=[]
    Ytrain=YtrainR
    bestauc=0.0
    end=len(X_train.columns)
    flag=0
    for i in range(0,end):
        Xtrain=X_train.iloc[:,:i+1]
        moduleDT = LogisticRegression(random_state=1)  
        moduleDT.fit(Xtrain, Ytrain)    
        _,train_tem=prob.CV_res(moduleDT,Xtrain,Ytrain,i,'True')#
        train_val.extend([train_tem])
        if train_tem[12]>=bestauc and i<=int(end/2):
            bestauc=train_tem[12]
            flag=i+1
    # np.savetxt(outpath+'\\'+outname+'_sfs.txt',np.array(train_val),delimiter='\t',fmt='%s')
    return flag,Xtrain.columns.tolist()
