# encoding=utf-8
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import scipy.io as scio
import joblib
import pandas as pd
import numpy as np
import os
data_path=os.path.split(os.path.realpath(__file__))[0]
clfname='xgb'
XFIS_test1=pd.read_csv(data_path.rsplit('\\',1)[0]+'/out/inte_data/COSMIC_testinter_prob.txt',sep='\t',header=None) 
data=pd.read_csv(data_path.rsplit('\\',1)[0]+'/out/ulti_data/iter_step5lr27_COSMIC'+'.txt',sep='\t',header=None)  
y_testR=data.iloc[:,0]
X_epSMic=data.iloc[:,[497,487,391,477,442,394,377,382,428,374,371,370,484,457,158,437]]
X_inter=XFIS_test1.iloc[:,1:]
fig = plt.figure(figsize=(13,4))

plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus']=False
ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
for j,name in zip([1,2],['X_inter','X_epSMic']):
        plt.subplot(1,2,j)
        y1=ts.fit_transform(eval(name))
        X1 =[]
        X0 =[]
        Y1 =[]
        Y0 =[]
        for i in range(X_epSMic.shape[0]):
                if y_testR[i]==1:
                        X1.append(y1[i,0])
                        Y1.append(y1[i,1])
                else:
                        X0.append(y1[i,0])
                        Y0.append(y1[i,1])
        a = plt.scatter(X1, Y1,color="orange")#hotpink
        b = plt.scatter(X0, Y0, color="slateblue")#deep;skyblue;mediumorchid
        if j==1:
                plt.legend((a, b),('Positive','Negative'), fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.axis('tight')
plt.plot()
plt.savefig(data_path+"/tsne-2test"+clfname+".png",dpi=600,bbox_inches="tight", pad_inches=0.2)
plt.close(0)
