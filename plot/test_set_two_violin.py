# encoding=utf-8
import os
import seaborn as sns
import numpy as np
# import torch
import os
from statannotations.Annotator import Annotator
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

data_path=os.path.split(os.path.realpath(__file__))[0]

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}
# 1.three tools
data_raw=pd.read_csv(data_path+'/testSet3.txt',sep='\t')
condition = data_raw['tool'].isin(['epSMic', 'CSS', 'CS'])
data_raw = data_raw[condition]
x='tool'
y='score'
hue='metrics'
order=['epSMic','CSS','CS']
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
fig, ax = plt.subplots(figsize = (8,6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax = sns.boxplot(data=data_raw, x=x, y=y,hue=hue,ax=ax, order=order,width=0.5) #
plt.yticks(np.arange(0.55,0.94,0.08),size = 30)
plt.xticks(size = 30, rotation=45)#
box_pairs= [ 
      (("epSMic","AUC"),("epSMic","AUPR"))
            ,(("CSS","AUC"),("CSS","AUPR")),     
             (("CS","AUC"),("CS","AUPR"))  
             ]
annotator = Annotator(ax, pairs=box_pairs, data=data_raw, x=x, y=y,hue=hue)
plt.legend(prop=font1)
plt.savefig(data_path+"/testSet2_3tools"+".png",dpi=600,bbox_inches="tight", pad_inches=0.2)
plt.close(0)

# 2.other tools
condition = data_raw['tool'].isin(['CSS', 'CS'])

data_raw = data_raw[~condition]

x='tool'
y='score'
hue='metrics'
order=['epSMic','PrDSM','TraP','PhD-SNPg','FATHMM-MKL','DANN','FATHMM-XF','EnDSM','frDSM','usDSM']#,
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
fig, ax = plt.subplots(figsize = (20,6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax = sns.boxplot(data=data_raw, x=x, y=y,hue=hue,ax=ax, order=order,width=0.5) #
plt.yticks(np.arange(0.35,0.94,0.08),size = 30)

plt.xticks(size = 30, rotation=45)#

box_pairs= [ 
      (("epSMic","AUC"),("epSMic","AUPR")),
             (("PrDSM","AUC"),("PrDSM","AUPR")),
             (("TraP","AUC"),("TraP","AUPR")),
             (("PhD-SNPg","AUC"),("PhD-SNPg","AUPR")),
             (("FATHMM-MKL","AUC"),("FATHMM-MKL","AUPR")),
             (("DANN","AUC"),("DANN","AUPR")),
             (("FATHMM-XF","AUC"),("FATHMM-XF","AUPR")),
             (("EnDSM","AUC"),("EnDSM","AUPR")),
             (("frDSM","AUC"),("frDSM","AUPR")),
             (("usDSM","AUC"),("usDSM","AUPR"))
            
             ]

annotator = Annotator(ax, pairs=box_pairs, data=data_raw, x=x, y=y,hue=hue) #, order=order
plt.legend(prop=font1)#,loc=9,ncol=2
plt.savefig(data_path+"/testSet2_8tools"+".png",dpi=600,bbox_inches="tight", pad_inches=0.2)
plt.close(0)

