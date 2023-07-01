# -*- coding: utf-8 -*-

from sklearn import metrics
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.utils import shuffle
def metr1_1(y_test,pred,pred_prob):
	tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()
	ACC=metrics.accuracy_score(y_test,pred)
	MCC=metrics.matthews_corrcoef(y_test,pred)
	recall=metrics.recall_score(y_test, pred)
	Pre = metrics.precision_score(y_test, pred) 
	specificity=tn/(tn+fp)
	BACC=(recall+specificity)/2.0
	F1_Score=metrics.f1_score(y_test, pred)
	fpr,tpr,threshold=metrics.roc_curve(y_test,pred_prob[:,1])
	roc_auc=metrics.auc(fpr,tpr)
	precision_prc, recall_prc, _ = metrics.precision_recall_curve(y_test, pred_prob[:,1])
	PRC = metrics.auc(recall_prc, precision_prc)
	return  [tn, fp, fn, tp,ACC,Pre,recall,MCC,specificity,BACC,F1_Score,roc_auc,PRC]
	
def fold2(moduleDT,X,y,CV=10):
	from sklearn.model_selection import StratifiedKFold
	from sklearn.base import clone
	skflods = StratifiedKFold(n_splits=CV,shuffle=False)
	a=[]
	y_pred_prob_all=np.zeros((X.shape[0],2))
	for train_index,test_index in skflods.split(X,y):
		clone_clf =clone(moduleDT)
		X_sub=np.array(X)
		y_sub=np.array(y)
		X_train_folds = X_sub[train_index]
		y_train_folds = y_sub[train_index]
		X_test_folds = X_sub[test_index]
		y_test_folds = y_sub[test_index]

		clone_clf.fit(X_train_folds,y_train_folds)
		y_pred = clone_clf.predict(X_test_folds)
		y_pred_prob = clone_clf.predict_proba(X_test_folds)
		y_pred_prob_all[test_index]=y_pred_prob
		a.append(metr1_1(y_test_folds,y_pred,y_pred_prob))
	a=np.array(a)
	return np.mean(a, axis=0),y_pred_prob_all