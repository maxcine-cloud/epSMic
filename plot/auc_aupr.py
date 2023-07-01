# encoding=utf-8
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os
data_path=os.path.split(os.path.realpath(__file__))[0]
font4 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   :  25
}
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 32,
}
def draw_roc(data, name,outpathway):
    data, nameList = data[:,:3],data[:,3]
    plt.subplot(1,2,1)
    colors = cycle(['#050f2c', 'goldenrod', '#00aeff', '#8e43e7', 'steelblue', '#ff6c5f', '#ffc168', '#2dde98','y','lightsteelblue','m','green','tan','blue','yellow']) 
    for i,(mol,color) in enumerate(zip(data,colors)):
        plt.plot(mol[0], mol[1], color=color,
                      lw=3 if i!=0 else 3, label=nameList[i]+' (AUC = %0.3f)' % mol[2], linestyle='dashdot' if i%2!=0 else '-')
    plt.xticks(size = 35,family= 'Times New Roman')
    plt.yticks(size = 35,family= 'Times New Roman')
    plt.xlim([-0.025, 1.075])
    plt.ylim([-0.025, 1.025])
    plt.xlabel('False Positive Rate',font3)
    plt.ylabel('True Positive Rate',font3)
    legend = plt.legend(loc="lower right",ncol= 1,prop=font3,columnspacing=0.2,labelspacing=0.2)  #lower right font4
    legend.get_title().set_fontsize(fontsize = 'small')

def draw_pr(data, name,outpathway):
    data, nameList = data[:,:3],data[:,3]
    plt.subplot(1,2,2)
    colors = cycle(['#050f2c', 'goldenrod', '#00aeff', '#8e43e7', 'steelblue', '#ff6c5f', '#ffc168', '#2dde98','y','lightsteelblue','m','green','tan','blue','yellow']) 
    for i,(mol,color) in enumerate(zip(data,colors)):
        plt.plot(mol[1], mol[0], color=color,
                      lw=3 if i!=0 else 3, label=nameList[i]+' (AUPR = %0.3f)' % mol[2], linestyle='dashdot' if i%2!=0 else '-')
    plt.xticks(size = 35,family= 'Times New Roman')
    plt.yticks(size = 35,family= 'Times New Roman')
    plt.xlim([-0.025, 1.025])
    plt.ylim([-0.025, 1.025])
    plt.xlabel('Recall',font3) 
    plt.ylabel('Precision',font3)    
    legend = plt.legend(loc="lower right",ncol= 1,prop=font3,columnspacing=0.2,labelspacing=0.2) #font4
    legend.get_title().set_fontsize('large')   # xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller, None 
    plt.savefig(outpathway+"\\au_step_4.1"+name+".png",dpi=600,bbox_inches="tight", pad_inches=0.2)
    plt.close(0)

type='test'
scores=pd.read_csv(data_path+'\\COSMIC_8tools.txt',sep='\t')
data_filter=scores.iloc[:,[4,5,6,7,8,9,10,11,12]]
data=np.array(data_filter).T.tolist()
a=np.loadtxt(data_path+'\\test_step5lr27_COSMIC.txt',dtype='str',delimiter='\t')[:,4]
data1=[]
data1.append(data[0])
data1.append(a.tolist())
new_data=np.row_stack([data1,data[1:]])#.tolist()
cs=np.loadtxt(data_path+'\\COSMIC_'+'CS'+'.txt',delimiter='\t',dtype=str,skiprows=1)
css=np.loadtxt(data_path+'\\COSMIC_'+'CSS'+'.txt',delimiter='\t',dtype=str,skiprows=1)
endsm=np.loadtxt(data_path+'\\COSMIC_'+'EnDSM'+'.txt',delimiter='\t',dtype=str,skiprows=1)
frdsm=np.loadtxt(data_path+'\\COSMIC_'+'frDSM'+'.txt',delimiter='\t',dtype=str,skiprows=1)
usdsm=np.loadtxt(data_path+'\\COSMIC_'+'usDSM'+'.txt',delimiter='\t',dtype=str,skiprows=1)
new_data=np.row_stack([new_data,np.array(css[:,4])])#.tolist(),,dtype='float64'
new_data=np.row_stack([new_data,np.array(cs[:,4])])#.tolist(),,dtype='float64'
# new_data=np.row_stack([new_data,np.array(synVep[:,4])])#.tolist(),,dtype='float64'
new_data=np.row_stack([new_data,np.array(endsm[:,4])])#.tolist(),,dtype='float64'
new_data=np.row_stack([new_data,np.array(frdsm[:,4])])#.tolist(),,dtype='float64'
new_data=np.row_stack([new_data,np.array(usdsm[:,4])])#.tolist(),,dtype='float64'
toolsnameList = ['epSMic','CSS','CS','SilVA','TraP','DANN','PrDSM','CADD','FATHMM-MKL','PhD-SNPg','FATHMM-XF','EnDSM','frDSM','usDSM'] #,'synVep','PredDSMC'
index=[1,10,11,4,3,8,2,7,6,5,9,12,13,14]
aucinputdata = []
praucinputdata = []
for i, other in enumerate(new_data[index]):#new_data[1:]
    y=new_data[0].reshape(-1,1)
    x=other.reshape(-1,1)
    sub=np.column_stack((y, x))
    sub[sub==''] = np.nan
    sub=np.array(sub,dtype='float32')
    sub=sub[~np.isnan(sub).any(axis=1), :]
    fpr, tpr, thresholds = roc_curve(sub[:,0],sub[:,1])#y_true
    _auc = auc(fpr, tpr)
    aucinputdata.append([fpr, tpr,_auc]) 
    precision, recall, _thresholds = precision_recall_curve(sub[:,0],sub[:,1])
    _prauc = auc(recall, precision)
    praucinputdata.append([precision, recall,_prauc])
newdata = np.column_stack([aucinputdata, toolsnameList])
plt.figure(figsize = (30,15)) #(36,18),(18,9)
draw_roc(newdata[[0,1,2],:], type,data_path)
# draw_roc(newdata[[0,3,4,5,6,7,8,9,10,11,12,13],:], type,data_path)
prnewdata = np.column_stack([praucinputdata, toolsnameList])
# draw_pr(prnewdata[[0,3,4,5,6,7,8,9,10,11,12,13],:], type,data_path)
draw_pr(prnewdata[[0,1,2],:], type,data_path)

