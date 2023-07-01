# -*- coding: utf-8 -*-

import numpy as np
from getVCF import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.svm import SVC
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import prob
# import featureSelect as sfs
import os
def iterUltimate_model(XFIS_test,idPath,outpath,modelpath,outname,iscosmic,istrain=0,interpath=''):
	test_val=[]
	modelLists=['ExtraTreesClassifier','GradientBoostingClassifier','LogisticRegression','RandomForestClassifier','AdaBoostClassifier'] 
	X_data=XFIS_test[:,1:]
	Y_data=XFIS_test[:,0]
	fc='5'
	for i in range(0,27,1):
		X_data_val=X_data
		for modelList in modelLists:
			print(i,modelList)
			if istrain==1:
				if os.path.exists(modelpath+'/'+modelList+str(i)+fc+'iter.model'):
					model = joblib.load(modelpath+'/'+modelList+str(i)+fc+'iter.model')
				else:
					if modelList=='SVC':
						model=eval(modelList)(random_state=1,probability=True)
					elif modelList=='KNeighborsClassifier'or 'MultinomialNB':
						model=eval(modelList)()
					else:model=eval(modelList)(random_state=1)
					model=model.fit(X_data_val, Y_data)
					joblib.dump(model,modelpath+'/'+modelList+str(i)+fc+'iter.model')
				predDT_proba,_=prob.CV_res(model,X_data_val,Y_data,i,'True')
				X_data=np.hstack((X_data,predDT_proba[:,1].reshape(-1,1)))
			else:
				model_last=joblib.load(modelpath+'/'+modelList+str(i)+fc+'iter.model')
				predDT_proba_t,_=prob.CV_res(model_last,X_data_val,Y_data,i)
				X_data=np.hstack((X_data,predDT_proba_t[:,1].reshape(-1,1)))
	# if istrain:
		# bestindex=sfs.clf_sfs(X_data,predDT_proba_t[:,1].reshape(-1,1),interpath)
	# np.savetxt(outpath+'/iter_step5lr27_'+outname+'.txt',np.hstack((np.array(Y_data).reshape(-1,1),X_data)),delimiter='\t',fmt='%s')  
	test=np.hstack((np.array(Y_data).reshape(-1,1),X_data))[:,[0,497,487,391,477,442,394,377,382,428,374,371,370,484,457,158,437]]

	test_val=[]
	X_test=test[:,1:]
	y_test=test[:,0]
	if istrain==1:
		if os.path.exists(modelpath+'/opti_after_'+modelLists[2]+'.model'):
			model_last = joblib.load(modelpath+'/opti_after_'+modelLists[2]+'.model')
		else:
			model_last=eval(modelLists[2])(random_state=1)
			model_last=model.fit(X_test, y_test)
			joblib.dump(model,modelpath+'/opti_after_'+modelLists[2]+'.model')
		predDT_proba_t,test_tem=prob.CV_res(model_last,X_test,y_test,0,'True')

	else:
		model_last=joblib.load(modelpath+'/opti_after_'+modelLists[2]+'.model')
		predDT_proba_t,test_tem=prob.CV_res(model_last,X_test,y_test,0)
	test_last=np.hstack((np.array(y_test).reshape(-1,1),np.round(predDT_proba_t[:,1],6).reshape(-1,1)))
	test_val.append(test_tem)
	test_val_arr=np.array(test_val)
	print("test:[ACC,Pre,recall,MCC,specificity,BACC,F1_Score,roc_auc,AUPR]=",test_val_arr[0,5:])
	if istrain==1:
		vcfid=getvcf(idPath,1,1)
		np.savetxt(outpath+'/train_step5lr27_'+outname+'.txt',np.hstack((vcfid,test_last[:,1:])),delimiter='\t',fmt='%s')
	else:
		if iscosmic==1:
			vcfid=getvcf(idPath,1)
		else:
			vcfid=getvcf(idPath)
		np.savetxt(outpath+'/test_step5lr27_'+outname+'.txt',np.hstack((vcfid,test_last[:,1:])),delimiter='\t',fmt='%s')	
	
	return test_val_arr

