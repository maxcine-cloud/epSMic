# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from pylab import *
import pandas as pd
import os
data_path=os.path.split(os.path.realpath(__file__))[0]

#1、three tools
names={'PrDSM':0.308,'TraP':0.459,'SilVA':0.278,'PhD-SNPg':0.5,'FATHMM-MKL':0.5,'CADD':15,'DANN':0.99,'FATHMM-XF':0.5}

data=[]
color_listts=[]
def add_index(epSMic_label):
    color_listt=[]
    lili={'0':'#050f2c','1':'orange','2':'#00aeff','3':'#8e43e7','4':'steelblue','5':'#ff6c5f','6':'#ffc168','7':'#2dde98','8':'y','9':'lightsteelblue','10':'m','11':'green','12':'tan','13':'blue'}
    for i in epSMic_label:
        if i ==1:
            j=1
        else:
            j=13      
        color_listt.append(lili[str(j)])
    color_listts.append(color_listt)  
valid=pd.read_csv(data_path+'/result_iter_step5_synony_valid.txt',sep='\t',header=None)
eosm=pd.read_csv(data_path+'\\test_step5lr27_EOSM.txt',sep='\t',header=None)
epSMic_vitro=eosm.iloc[:,4]
labell=valid.iloc[:,12].tolist()
data.append(np.array(epSMic_vitro).reshape(58,))
add_index(labell)


    
a={'CSS':4,'CS':4}
for key,value in a.items():
    if key == 'CSS' :
        tool=pd.read_csv(data_path+'/CSSscore_test_valid.txt',sep='\t').iloc[:,[value,value+3]]
    else:
        tool=pd.read_csv(data_path+'/CSscore_test_valid.txt',sep='\t').iloc[:,[value,value+3]]
    inde=np.where(~tool.iloc[:,0].isna())
    add_index(tool.iloc[inde[0].tolist(),1])
    tool=tool.iloc[:,0].dropna()
    
    tool=np.array(tool)
    data.append(tool)
    

species = ['epSMic','CSS','CS'] 

y_data = data


jitter = 0.04
x_data = [np.array([i] * len(d)) for i, d in enumerate(y_data)]
x_jittered = [x + st.t(df=6, scale=jitter).rvs(len(x)) for x in x_data]


BG_WHITE = "#fbf9f4"
GREY_LIGHT = "#b4aea9"
GREY50 = "#7F7F7F"
BLUE_DARK = "#1B2838"
BLUE = "#2a475e"
BLACK = "#282724"
GREY_DARK = "#747473"
RED_DARK = "#850e00"
POSITIONS = [0,1,2]
fig, ax = plt.subplots(figsize= (10, 6))

HLINES = [-0.10,0.10,0.30,0.50,0.70,0.90]
for h in HLINES:
    ax.axhline(h, color=GREY50, ls=(0, (5, 5)), alpha=0.8, zorder=0)
violins = ax.violinplot(
    y_data, 
    positions=POSITIONS,
    widths=0.75,
    bw_method="silverman",
    showmeans=False, 
    showmedians=False,
    showextrema=False
)
for pc in violins["bodies"]:
    pc.set_facecolor("aliceblue")
    pc.set_edgecolor(BLACK)
    pc.set_linewidth(1.4)
    pc.set_alpha(1)

medianprops = dict(
    linewidth=4, 
    color=GREY_DARK,
    solid_capstyle="butt"
)
boxprops = dict(
    linewidth=2, 
    color=GREY_DARK
)

ax.boxplot(
    y_data,
    positions=POSITIONS, 
    showfliers = False,
    showcaps = False,
    medianprops = medianprops,
    whiskerprops = boxprops,
    boxprops = boxprops,
    widths = 0.2
)

COLOR_SCALE = ['#050f2c', 'goldenrod', '#00aeff', '#8e43e7', 'steelblue', '#ff6c5f', '#ffc168', '#2dde98','y','lightsteelblue','m','green','tan','blue','yellow']#["#1B9E77", "#D95F02", "#7570B3"]
for x, y, color,labell in zip(x_jittered, y_data, COLOR_SCALE,color_listts):
    ax.scatter(x, y, s = 110,  alpha=0.4,c=labell)
means = [y.mean() for y in y_data]
for i, mean in enumerate(means):
    ax.scatter(i, mean, s=250, color=RED_DARK, zorder=3)
    ax.plot([i, i + 0.25], [mean, mean], ls="dashdot", color="black", zorder=3)
    ax.text(
        i + 0.25,
        mean,
        r"$\hat{\mu} = $" + str(round(mean, 2)),
        fontsize=15,
        fontname="Times New Roman",
        va="center",
        bbox = dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round",
            pad=0.15
        ),
        zorder=10
    )
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")

ax.spines["left"].set_color(GREY_LIGHT)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_color(GREY_LIGHT)
ax.spines["bottom"].set_linewidth(2)


ax.tick_params(length=0)
ax.set_yticks(HLINES)
ax.set_yticklabels(HLINES, size=18,fontname="Times New Roman")
ax.set_ylabel("Effect Score", size=20, weight="bold",fontname="Times New Roman")


xlabels = [f"{specie}" for i, specie in enumerate(species)]
ax.set_xticks(POSITIONS)
ax.set_xticklabels(xlabels, size=17, ha="center", ma="center",fontname="Times New Roman")
ax.set_xlabel("Functional Tool", size=18, weight="bold",fontname="Times New Roman")

plt.savefig(data_path+'/box_valid_label_3tools2'+'.png', transparent=True, dpi=600, bbox_inches="tight", pad_inches=0.2)


#2、other tools
names={'PrDSM':0.308,'TraP':0.459,'SilVA':0.278,'PhD-SNPg':0.5,'FATHMM-MKL':0.5,'CADD':15,'DANN':0.99,'FATHMM-XF':0.5}
data=[]
color_listts=[]
def add_index(epSMic_label):
    color_listt=[]
    lili={'0':'#050f2c','1':'orange','2':'#00aeff','3':'#8e43e7','4':'steelblue','5':'#ff6c5f','6':'#ffc168','7':'#2dde98','8':'y','9':'lightsteelblue','10':'m','11':'green','12':'tan','13':'blue'}
    for i in epSMic_label:
        if i ==1:
            j=1
        else:
            j=13      
        color_listt.append(lili[str(j)])
    color_listts.append(color_listt)    
valid=pd.read_csv(data_path+'/result_iter_step5_synony_valid.txt',sep='\t',header=None)
eosm=pd.read_csv(data_path+'/test_step5lr27_EOSM.txt',sep='\t',header=None)
epSMic_vitro=eosm.iloc[:,4]

labell=valid.iloc[:,12].tolist()
data.append(np.array(epSMic_vitro).reshape(58,))
add_index(labell)

scores=pd.read_csv(data_path+'/score_test_valid.txt',sep='\t')
data_filter=scores.iloc[:,4:13]
for i,j in zip(range(0,8,1),names):
    if i==5 or i==2:
        continue
    sub=np.array(data_filter.iloc[:,i]).reshape(58,)
    inde=np.where(~np.isnan(sub))
    add_index(data_filter.iloc[inde[0].tolist(),8])
    sub=sub[~np.isnan(sub)]
    sub=sub-names[j]+0.5
    data.append(sub)
    
    
a={'EnDSM':5,'frDSM':5,'usDSM':5}


for key,value in a.items():
    tool=pd.read_csv(data_path+'/test_valid_'+key+'.csv',sep=',').iloc[:,[value,value+1]]
    inde=np.where(~tool.iloc[:,0].isna())
    add_index(tool.iloc[inde[0].tolist(),1])
    tool=tool.iloc[:,0].dropna()
    
    tool=np.array(tool)
    data.append(tool)

species = ['epSMic','PrDSM','TraP','PhD-SNPg','FATHMM-MKL','DANN','FATHMM-XF','EnDSM','frDSM','usDSM']
y_data = data


jitter = 0.04
x_data = [np.array([i] * len(d)) for i, d in enumerate(y_data)]
x_jittered = [x + st.t(df=6, scale=jitter).rvs(len(x)) for x in x_data]


BG_WHITE = "#fbf9f4"
GREY_LIGHT = "#b4aea9"
GREY50 = "#7F7F7F"
BLUE_DARK = "#1B2838"
BLUE = "#2a475e"
BLACK = "#282724"
GREY_DARK = "#747473"
RED_DARK = "#850e00"
POSITIONS = [0,1,2,3,4,5,6,7,8,9]
fig, ax = plt.subplots(figsize= (27,10))

HLINES = [-0.10,0.10,0.30,0.50,0.70,0.90]
for h in HLINES:
    ax.axhline(h, color=GREY50, ls=(0, (5, 5)), alpha=0.8, zorder=0)
violins = ax.violinplot(
    y_data, 
    positions=POSITIONS,
    widths=0.75,
    bw_method="silverman",
    showmeans=False, 
    showmedians=False,
    showextrema=False
)
for pc in violins["bodies"]:
    pc.set_facecolor("aliceblue")
    pc.set_edgecolor(BLACK)
    pc.set_linewidth(1.4)
    pc.set_alpha(1)

medianprops = dict(
    linewidth=4, 
    color=GREY_DARK,
    solid_capstyle="butt"
)
boxprops = dict(
    linewidth=2, 
    color=GREY_DARK
)

ax.boxplot(
    y_data,
    positions=POSITIONS, 
    showfliers = False,
    showcaps = False,
    medianprops = medianprops,
    whiskerprops = boxprops,
    boxprops = boxprops,
    widths = 0.2
)

COLOR_SCALE = ['#050f2c', 'goldenrod', '#00aeff', '#8e43e7', 'steelblue', '#ff6c5f', '#ffc168', '#2dde98','y','lightsteelblue','m','green','tan','blue','yellow']#["#1B9E77", "#D95F02", "#7570B3"]
for x, y, color,labell in zip(x_jittered, y_data, COLOR_SCALE,color_listts):
    ax.scatter(x, y, s = 110,  alpha=0.4,c=labell)


means = [y.mean() for y in y_data]
for i, mean in enumerate(means):
    ax.scatter(i, mean, s=250, color=RED_DARK, zorder=3)
    ax.plot([i, i + 0.25], [mean, mean], ls="dashdot", color="black", zorder=3)
    ax.text(
        i + 0.25,
        mean,
        r"$\hat{\mu} = $" + str(round(mean, 2)),
        fontsize=15,
        fontname="Times New Roman",
        va="center",
        bbox = dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round",
            pad=0.15
        ),
        zorder=10
    )

ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")

ax.spines["left"].set_color(GREY_LIGHT)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_color(GREY_LIGHT)
ax.spines["bottom"].set_linewidth(2)

ax.tick_params(length=0)
ax.set_yticks(HLINES)
ax.set_yticklabels(HLINES, size=18,fontname="Times New Roman")
ax.set_ylabel("Effect Score", size=20, weight="bold",fontname="Times New Roman")

xlabels = [f"{specie}" for i, specie in enumerate(species)]
ax.set_xticks(POSITIONS)
ax.set_xticklabels(xlabels, size=17, ha="center", ma="center",fontname="Times New Roman")
ax.set_xlabel("Functional Tool", size=18, weight="bold",fontname="Times New Roman")

plt.savefig(data_path+'/box_valid_label_othertools2'+'.png', transparent=True, dpi=600, bbox_inches="tight", pad_inches=0.2)