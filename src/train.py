# -*- coding: utf-8 -*-

import numpy as np
import iterUltimate
import intermedium
from arg import *
from getVCF import *
from dataProcess import *
from loadfile import *


def train1():
	args=get_args()
	feat_name=['conservation','sequence','score','splicing','embedding2','diffe_feature2mer','diffe_featureCKSNAP','diffe_featureMismatch','diffe_featureNAC','diffe_featureRC2mer','diffe_featureMMI','diffe_featureZ_curve_9bit','diffe_featureZ_curve_12bit','diffe_featureZ_curve_36bit','diffe_featureZ_curve_48bit','diffe_featureZ_curve_144bit','diffe_featureNMBroto','mutation_2mer','mutation_CKSNAP','mutation_Mismatch','mutation_NAC','mutation_RC2mer','mutation_MMI','mutation_Z_curve_9bit','mutation_Z_curve_12bit','mutation_Z_curve_36bit','mutation_Z_curve_48bit','mutation_Z_curve_144bit','mutation_NMBroto','normal_2mer','normal_CKSNAP','normal_Mismatch','normal_NAC','normal_RC2mer','normal_MMI','normal_Z_curve_9bit','normal_Z_curve_12bit','normal_Z_curve_36bit','normal_Z_curve_48bit','normal_Z_curve_144bit','normal_NMBroto']
	test_procedList=[]
	for i in feat_name:
		test_i=im_file(args.dataPath+'/'+args.dbName+'/'+i,1,'training')
		Xtest_i=test_i
		unique2=getvcf(args.dataPath+'/'+args.dbName+'/'+args.dbName,1,1,0)
		print(i)  
		Xtest_proced=dataProcessing(Xtest_i, args.processingmodelPath+'/'+i,1)
		test_proced=np.hstack((unique2,Xtest_proced))
		test_procedList.append(test_proced)
  
	iter_prob=intermedium.interme_model(test_procedList,test_i.shape[0],args.intermodelPath,1,1,interdataName=args.interdataPath+'/'+args.dbName+'_'+args.dataType)
	test_val_arr=iterUltimate.iterUltimate_model(iter_prob,args.dataPath+'/'+args.dbName+'/'+args.dbName,args.processedPath,args.itermodelPath,args.dbName,1,1,args.interdataPath)




