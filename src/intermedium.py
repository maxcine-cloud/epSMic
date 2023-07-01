# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import json
import os
from sklearn.model_selection import StratifiedKFold
import joblib
import prob

def interme_model(data_tList,nnum1,modelpath,islabel,istrain=0,interdataName=''):
	feat_name=['conservation','sequence','score','splicing','embedding2','diffe_feature2mer','diffe_featureCKSNAP','diffe_featureMismatch','diffe_featureNAC','diffe_featureRC2mer','diffe_featureMMI','diffe_featureZ_curve_9bit','diffe_featureZ_curve_12bit','diffe_featureZ_curve_36bit','diffe_featureZ_curve_48bit','diffe_featureZ_curve_144bit','diffe_featureNMBroto','mutation_2mer','mutation_CKSNAP','mutation_Mismatch','mutation_NAC','mutation_RC2mer','mutation_MMI','mutation_Z_curve_9bit','mutation_Z_curve_12bit','mutation_Z_curve_36bit','mutation_Z_curve_48bit','mutation_Z_curve_144bit','mutation_NMBroto','normal_2mer','normal_CKSNAP','normal_Mismatch','normal_NAC','normal_RC2mer','normal_MMI','normal_Z_curve_9bit','normal_Z_curve_12bit','normal_Z_curve_36bit','normal_Z_curve_48bit','normal_Z_curve_144bit','normal_NMBroto']#feature groups compare
	modelList=['SVC','RandomForestClassifier','KNeighborsClassifier','DecisionTreeClassifier','MultinomialNB','LogisticRegression','ExtraTreesClassifier','GradientBoostingClassifier','AdaBoostClassifier']
	iter_prob=np.ones((nnum1,))
	for i,count in zip(feat_name,range(0,len(data_tList))):
		data_t=data_tList[count]
		if islabel==1:
			X_t=data_t[:,5:]
			y_t=data_t[:,4]	
		else:
			X_t=data_t[:,4:]
			y_t=np.zeros((nnum1,))      
		if istrain==1:
			y_t=y_t.astype('int')
			for countd in range(len(modelList)):
				print(i,modelList[countd])
				if os.path.exists(modelpath+'/'+i+'_'+modelList[countd]+'.model'):
					model_base=joblib.load(modelpath+'/'+i+'_'+modelList[countd]+'.model')
				else:
					if modelList[countd]=='SVC':
						model=eval(modelList[countd])(random_state=1,probability=True)
					elif modelList[countd]=='KNeighborsClassifier'or 'MultinomialNB':
						model=eval(modelList[countd])()
					else:model=eval(modelList[countd])(random_state=1)
					model_base=model.fit(X_t, y_t)
					joblib.dump(model_base,modelpath+'/'+i+'_'+modelList[countd]+'.model')
				countd=countd+1
				predDT_proba,_=prob.CV_res(model_base,X_t,y_t,i,'True')
				iter_prob=np.vstack((iter_prob,predDT_proba[:,1])) 			
		else:
			for j in modelList:
				model_base=joblib.load(modelpath+'/'+i+'_'+j+'.model')
				base_proba = model_base.predict_proba(X_t)
				iter_prob=np.vstack((iter_prob,base_proba[:,1]))
	if islabel==1:
		iter_prob[0,:]=y_t
	# np.savetxt(interdataName+'inter_prob.txt', iter_prob.T,delimiter='\t',fmt='%s')
	return iter_prob.T
	
