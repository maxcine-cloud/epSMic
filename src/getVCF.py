# -*- coding: utf-8 -*-
import numpy as np
def getvcf(idName,iscosmic=0,istrain=0,onlyid=1):
    if onlyid == 1:
        if iscosmic==1:
            if istrain ==1:
                return np.loadtxt(idName+'_training.vcf',dtype=str,delimiter='\t',skiprows=1)[:,:4]
            else:
                return np.loadtxt(idName+'_testing.vcf',dtype=str,delimiter='\t',skiprows=1)[:,:4]
        else:
            return np.loadtxt(idName+'.vcf',dtype=str,delimiter='\t',skiprows=1)[:,:4]
    else:
        if iscosmic==1:
            if istrain ==1:
                return np.loadtxt(idName+'_training.vcf',dtype=str,delimiter='\t',skiprows=1)
            else:
                return np.loadtxt(idName+'_testing.vcf',dtype=str,delimiter='\t',skiprows=1)
        else:
            return np.loadtxt(idName+'.vcf',dtype=str,delimiter='\t',skiprows=1)    