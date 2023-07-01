# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.pyplot import MultipleLocator
import matplotlib as mpl

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 22,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}

data_path=os.path.split(os.path.realpath(__file__))[0]
data=pd.read_csv(data_path+'/iter_num.txt',sep='\t')
datalists=[]
label=data.columns.tolist()
for i in range(0,5):
    datalists.append(np.array(data)[:,i])   
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
names= ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30'] #x轴标题
plt.figure(figsize = (19,12))
plt.ylim(0.823, 0.846) 
plt.yticks(np.arange(0.823,0.846,0.002))
markers=['o','.','^','v','2','*','s','+']
x=names
for name,y,mar in zip(label,datalists,markers):
    plt.plot(x, y,  marker='o' ,label=name,markersize=9,linewidth=3)
plt.legend(prop=font2) 
plt.xticks(x, names,size = 35)
plt.yticks(size = 35)
plt.subplots_adjust(bottom=0.15)
ax = plt.gca()
plt.xlabel('Iterations',font1)
plt.ylabel("ACC",font1)
plt.margins(0.01)
tick_spacing = 3
ax.xaxis.set_major_locator(mtick.MultipleLocator(tick_spacing))
plt.savefig(data_path+"/iter_num_new"+".png",dpi=600,bbox_inches="tight", pad_inches=0.2)
plt.close(0)