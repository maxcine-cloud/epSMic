# -*- coding: utf-8 -*-
from sfs import *
from rank_tree import *
from rank_mrmr import *
import pandas as pd

def clf_sfs(X_train,YtrainR,outpath):
    rankfc='XGBClassifier'
    # featurn_num,all_bestindex=SFS(rank('GradientBoostingClassifier',X_train,YtrainR),YtrainR,outpath,'gbdt')
    featurn_num,all_bestindex=SFS(rank('XGBClassifier',X_train,YtrainR),YtrainR,outpath,'xgb')
    # featurn_num,all_bestindex=SFS(rank('RandomForestClassifier',X_train,YtrainR),YtrainR,outpath,'rf')
    # featurn_num,all_bestindex=SFS(rankmrmr('mrmr',X_train,YtrainR),YtrainR,outpath,'mrmr')
    bestindex=['0']+all_bestindex[:featurn_num]
    print(bestindex)
    return bestindex

