# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from pylab import *
import pandas as pd
import os
font5 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}
name=[]
for i in range(1,505):
    name.append(i)
path=os.path.split(os.path.realpath(__file__))[0]
y=pd.read_csv(path+'/train_gbdt_auc.txt',sep='\t',header=None)
y[1] = y[1].astype(str)
y1=y.iloc[:,0].tolist()
x=name
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
plt.figure(figsize = (40,10))
plt.ylim(0.905, 0.914)
plt.plot(x, y1, marker='o', mec='r', mfc='w',markersize=2,linewidth=5)
xticks=list(range(0,len(x),18))
xlabels=[x[x1] for x1 in xticks]
xticks.append(len(x))
xlabels.append(x[-1])
plt.xticks(xlabels, xlabels, rotation=45,size = 30)
plt.yticks(size = 30)
plt.margins(0)
plt.xlabel('Feature Number',font5)
plt.ylabel("AUC",font5)
plt.margins(0.005)
plt.savefig(path+"/sfs_rank3"+".png",dpi=600,bbox_inches="tight", pad_inches=0.2)
plt.close(0)