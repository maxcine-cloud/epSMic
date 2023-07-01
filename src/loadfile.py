# -*- coding: utf-8 -*-
import pandas as pd

def im_file(name,iscosmic=0,type=''):
	if iscosmic==1:
		return pd.read_csv(name+'_'+type+'.txt',sep='\t')
	else:
		return pd.read_csv(name+'.txt',sep='\t')

