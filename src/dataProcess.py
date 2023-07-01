# -*- coding: utf-8 -*-

import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os

def dataProcessing(data,i,istrain=0):
	if istrain==1:
		if os.path.exists(i+'.dataProcessing'):
			clf=joblib.load(i+'.dataProcessing')
			clf.transform(data)
		else:
			clf = Pipeline([('missing_values',SimpleImputer(missing_values=np.nan,strategy='mean')), ('minmax_scaler',MinMaxScaler())]) 
			clf.fit(data)
			joblib.dump(clf,i+'.dataProcessing')
	else:
		clf=joblib.load(i+'.dataProcessing')
	return clf.transform(data)
