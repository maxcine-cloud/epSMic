# -*- coding: utf-8 -*-
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier      
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

def rank(ways,XtrainR,YtrainR):
    clf=eval(ways)(random_state = 1).fit(XtrainR, YtrainR)
    importance=clf.feature_importances_
    Impt_Series=pd.Series(importance,index=XtrainR.columns)
    Impt_Series.sort_values(ascending=True).plot(kind='barh')
    index=pd.Series(importance,index=[a for a,_  in zip(range(0,len(XtrainR.columns)),XtrainR.columns)]).sort_values(ascending=False).index.tolist()
    X_train=XtrainR.iloc[:,index]
    print(X_train.columns)
    # np.savetxt('./out/inte_data/'+ways+'rank_colunms.txt',np.array(X_train.columns.tolist()),delimiter='\t',fmt='%s')
    return X_train
