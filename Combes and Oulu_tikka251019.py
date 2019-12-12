

#Import and analyse experimental data, 7 phases, by Pauli Tikka, 10.12.2019

#%%1) IMPORTING CALCULATION AND OPERATION PACKAGES
import pandas as pd #for importing files
import numpy as np  #for calculations, array manipulations, and fun :)
import matplotlib.pyplot as plt #for scientifical plots
import matplotlib as mpl
import random #for randomizing indeces
import pylab #correct x and y limits after plotting
import statsmodels.api as sm #for statistic models
from random import randrange, uniform #uniform function makes random float values between a range
from scipy import stats #for statistics
from scipy import * #everything from scipy
from scipy.stats import ks_2samp # comparison of two samples
from statsmodels.api import qqplot_2samples #for QQ plots
from numpy import * # e.g isnan command
import itertools
import statistics
import os
import matplotlib.colors as clr
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

#%% 2) MAKING OF THE FUNCTIONS FOR THE CALCULATIONS
def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    
        return m, m-h, m+h    

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def convertTuple(tup): 
    str =  ''.join(tup) 
    return str
    bxan=ravel(bxan)
    bxan=np.delete(bxan,obj=1)
    bxana=[]
    for i in range(len(bxan)):
        bxana.append(bxan[i]-2*(bxan[i]-3689))

def corr2hex(n):
    ''' Maps a number in [0, 1] to a hex string '''
    if n == 1: return '#fffff'
    else: return '#' + hex(int(n * 16**6))[2:].zfill(6)

def model_to_semi_som(df_pos_np3d):
    time=len((df_pos_np3d.set_index("time_mcs")).index.unique())
    dfoki2=[]
    dfi =[]
    dfoki2=df_pos_np3d.set_index("time_mcs").set_index("cell_no")
    criteria=dfcell_names3da[1]=='NPCells'
    listi=dfcell_names3da[0][criteria]
    dfooo=dfoki2.ix[np.array(listi)]
    indeces=np.array(listi)
    index=0
    mmf=[]
    mmf2=[]
    for index in indeces:
        mmf.append(np.array(dfooo.ix[index,0:3]))  #NP all coordinates
    time=2000 
    cells=392 #
    inst_speed_np3d=inst_fun(df_pos_np3d,time,cells) 
    criteria2=dfcell_names3da[1]!='Wall'
    listia=dfcell_names3da[0][criteria2]
    len(listia)
    listia=pd.DataFrame(listia)
    listia.index = range(392)
    #https://stackoverflow.com/questions/19609631/python-changing-row-index-of-pandas-data-frame
    #https://www.geeksforgeeks.org/python-intersection-two-lists/
    listia.loc[listia[0].isin(listi)] 
    npt=listia.loc[listia[0].isin(listi)].index
    speed_np3d=inst_speed_np3d[npt]
    speeds_np3d=np.mean(speed_np3d,axis=1)
    #https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
    mmf2=np.mean(mmf, axis=1)    #One cell's mean coordinate during experiemnts  
    N = len(mmf2[:,0])
    x = mmf2[:,0]/max(mmf2[:,0])
    y = mmf2[:,1]/max(mmf2[:,1])  
    for i in range(len(y)):
        if y[i] >= 0.95:
            y[i]=0.5
        if y[i] <= 0.082:
            y[i]=0.1   
    z = mmf2[:,2]/max(mmf2[:,2])
    colors = speeds_np3d/max(speeds_np3d) 
    colors=np.round(colors,1) 
    #oki
    area = (30 * z)**2  
    return x, y, x, colors, area
	
#http://numpy-discussion.10968.n7.nabble.com/Rolling-window-moving-average-moving-std-and-more-td4744.html
def rolling_window(a, window): 
    """ 
    Make an ndarray with a rolling window of the last dimension 
    Parameters 
    ---------- 
    a : array_like 
        Array to add rolling window to 
    window : int 
        Size of rolling window 
    Returns 
    ------- 
    Array that is a view of the original array with a added dimension 
    of size w. 
    Examples 
    -------- 
    >>> x=np.arange(10).reshape((2,5)) 
    >>> rolling_window(x, 3) 
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]], 
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]]) 
    Calculate rolling mean of last dimension: 
    >>> np.mean(rolling_window(x, 3), -1) 
    array([[ 1.,  2.,  3.], 
           [ 6.,  7.,  8.]]) 
    """ 
    if window < 1: 
        raise( ValueError, "`window` must be at least 1.") 
    if window > a.shape[-1]: 
        raise (ValueError, "`window` is too long.") 
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window) 
    strides = a.strides + (a.strides[-1],) 
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

#%% 3) IMPORTING EXPERIMENTAL FILES 
# Combes et al. 2016 excel files:
dfc_experim = pd.read_excel('C:/python/' + 'mmc5.xlsx')
dfc_experim2 = pd.read_excel('C:/python/' + 'mmc4.xlsx')
# Saarela et al. 2017, these are the nephron progenitor cell speeds of the Oulu experiments (2016/17)
df1_experimi = pd.read_excel('C:/python/' + 'right-speed.xls', header=None)
df2_experimi = pd.read_excel('C:/python/' + 'left-speed.xls', header=None)
df3_experimi = pd.read_excel('C:/python/' + 'Whole UB tip-speed.xls', header=None)
#%% 3/2018 oulu experiment recut from 7b:
df3_3d7b_speed = pd.read_csv('C:/python/speed_7b3d.csv', delimiter=',', header=None) #or
df_speed_ilya7b = pd.read_csv('C:/python/speed_7b3d.csv', delimiter=',', header=None) 
#% 3/2018 oulu experiment recut from 5b, also 0.24
df3_5b3d_speed = pd.read_csv('C:/python/speed_5b3d.csv', delimiter=',', header=None)
#%%Oulu Data
# Oulu 2018 positions->distances, always check the header
df_oulu_x = pd.read_csv('C:/python/position_1_7b3d.csv', delimiter=',', header=None)
df_oulu_y = pd.read_csv('C:/python/position_2_7b3d.csv', delimiter=',', header=None)
df_oulu_z = pd.read_csv('C:/python/position_3_7b3d.csv', delimiter=',', header=None)

#%% 4) MAKING OF THE REFERENCE SETS
#You may also use all the samples:
dfce2_v3=dfc_experim2[['sample','time','speed','y']]
dfce2_v3=dfce2_v3[95:]
dfce2_v3=dfce2_v3.dropna(axis=0) #note, axis=0
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html
lot=list(range(2,50))
#https://cmdlinetips.com/2018/02/how-to-subset-pandas-dataframe-based-on-values-of-a-column/
dfce2_v3=dfce2_v3[dfce2_v3.time.isin(lot)]
dfce2_v3['speed']=dfce2_v3['speed']*60
dfz2=dfce2_v3.set_index("sample")
dfz2=dfce2_v3[dfce2_v3['sample']==4] #riittää määrää
dfz2['y']
yn = [(float(i)-min(dfz2['y']))/(max(dfz2['y'])-min(dfz2['y'])) for i in dfz2['y']]
dfz2['yn']=yn
dfz2yn=dfz2[dfz2['yn']>0.6] 
dfz2yn['time'].nunique()
ok=dfz2yn['time'].value_counts()
heihou=ok.sort_index()
hh2=heihou*(150/np.max(heihou))
plt.plot(hh2)
hh3=hh2[8:]  
yes=[]
yes2=[]
yes3=[]
#%%
for i in range(time):
    yes.append(mean_confidence_interval(result[[i]], confidence=0.95))
dfz=dfce2_v3.set_index("sample")
yn = [(float(i)-min(dfz['y']))/(max(dfz['y'])-min(dfz['y'])) for i in dfz['y']]
dfz['yn']=yn
dfz2yn=dfz[dfz['yn']>0.3818340295]     # iteroidaan x (0.1-0.5): Okt: 0.1,0.2,0.3->0.4 ei ok 
len(dfz[dfz['yn']>0.3818340295].index.unique()) #tulis olla 8
dfz2yn=pd.DataFrame(dfz2yn)
index = 0
ind=[4, 6, 11, 13, 14, 16, 17]
ok=[]
for index in ind:
    ok.append(dfz2yn['time'].ix[index].value_counts())

ok[3] = np.concatenate(ok[3].insert(31,1) 
#%%
heihou=[]
hh2=[]
for i in range(len(ok)):    
    heihou.append(ok[i].sort_index())

    hh2.append(heihou[i]*(125/np.max(heihou[i]))) 
    
        dfJEE=dataframes.set_index("cell_no")

    index = 0
    mmf=[]
    for index in setti: 
        mmf.append(np.array(dfJEE.ix[index,
matrix = []
for i in range(len(hh2)):
    matrix.append([hh2[i], hh2[i].index])
hh3=np.array(hh2) #important
#https://cognitiveclass.ai/blog/nested-lists-multidimensional-numpy-arrays
hh4=hh3.T
yes=[]
for i in range(len(hh4)):
    yes.append(mean_confidence_interval(hh4[i], confidence=0.95))                                    
yes=pd.DataFrame(yes) 
for i in range(len(yes)):
    yes.loc[i,1]=float(yes.loc[i,1])
    yes.loc[i,2]=float(yes.loc[i,2])
yes[1] = yes[1].astype(float)
yes[2] = yes[2].astype(float) 
yes2=np.repeat(yes, 2)
yes2=pd.DataFrame()
yes2[0]=np.repeat(yes.loc[:,0], 2)
yes2[1]=np.repeat(yes.loc[:,1], 2)
yes2[2]=np.repeat(yes.loc[:,2], 2)
jaa=yes[::-1]    
jaa2=list(jaa[0])
jaa2=np.repeat(jaa2, 2)    
yes3=pd.DataFrame(jaa2)   
yes3.to_csv('C:\python\yes2',index=False,header='infer') #Tallennus samassa     
yes3=pd.DataFrame(hhn2)  
yes3.to_csv('C:\python\yesi',index=False,header='infer') 
#https://stackoverflow.com/questions/24225072/repeating-elements-of-a-list-n-times/24225106  
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html
#https://cognitiveclass.ai/blog/nested-lists-multidimensional-numpy-arrays
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_index.html
#https://stackoverflow.com/questions/6422700/how-to-get-indices-of-a-sorted-array-in-python
#https://stackoverflow.com/questions/38309729/count-unique-values-with-pandas-per-groups/38309823                           
ok=dfz2yn['time'].value_counts()
heihou=ok.sort_index()
hh2=heihou*(150/np.max(heihou))       
dfz2=dfz2.set_index("time")
#% Select randomly from explant culture and model data
combes_speed=np.random.choice(dfz2['speed'], 2000, False)
model_speed=np.random.choice(np.concatenate(test), 2000, False)
df_testt2=[]
df_col=list(range(0,2001))
df_testt2=pd.DataFrame(model_speed)
df_testt2["time"]=pd.DataFrame(df_col)
df_testt2["experim"]=pd.DataFrame(combes_speed)
df_testt2.columns = ["model","time","experim"]
columnsTitles=["time","model","experim"]
df_testt2=df_testt2.reindex(columns=columnsTitles)
#Save the data set as a reference:
df_testt2.to_csv('C:/python/a reference set_speed_test.csv', header=True, index=None )
dfz22=dfz2['speed']
index=0
iittas=[]
piittas=[]
for index in list(lot):
    iittas.append(np.array(dfz22[index].sample(n=22)))
piittas=[]    
for index in list(lot):    
    piittas.append(dfz22[index].iloc[(dfz22[index]-np.mean(dfz22[index])).abs().argsort()[:22]])
ypp = np.concatenate([np.array(i) for i in piittas])
ypp=ypp[0:1000]
sort(dfz22[49].iloc[(dfz22[49]-np.mean(dfz22[49])).abs().argsort()[:22]])
#%% Combes tipEndDistances
#https://stackoverflow.com/questions/10346336/list-of-lists-into-numpy-array
dfce2_v3=dfc_experim2[['sample','time','tipendDist']] 
dfce2_v3=dfce2_v3.dropna(axis=0) #note, axis=0  
dfz2=dfce2_v3[dfce2_v3['sample']==4] #riittää määrää
dfz2=dfz2.set_index("time")
dfz22=dfz2['tipendDist']
combes=np.random.choice(dfz22, 3500, False)
A2x=np.concatenate(A2)
A2xx=list(np.concatenate(A2x))
#group_of_items = {1, 2, 3, 4}               # a sequence or set will work here.
num_to_select = 3500                           # set the number to select here.
list_of_random_items = random.sample(A2xx, num_to_select)
ub3d_adh_corn=np.array(list_of_random_items)
df_testt2=[]
df_col=list(range(0,3501))
df_testt2=pd.DataFrame(ub3d_adh_corn)
df_testt2["time"]=pd.DataFrame(df_col)
df_testt2["experim"]=pd.DataFrame(combes)
df_testt2.columns = ["model","time","experim"]
columnsTitles=["time","model","experim"]
df_testt2=df_testt2.reindex(columns=columnsTitles)
# A reference set with more values
df_testt2.to_csv('C:/python/a reference set_ub3d_adh2_tot_ok.csv', header=True, index=None )
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html
lot=list(range(2,50))
index=0
iittas=[]
for index in list(lot):
    iittas.append(np.array(dfz22[index].sample(n=22)))
yp = np.concatenate([np.array(i) for i in iittas]) #does the flattening
yp=yp[0:4999]
plt.plot(yp)
yp2=ypp[0:300]
#https://pandas.pydata.org/pandas-docs/stable/reshaping.html
df_col=list(range(0,301))
df_testt2=[]
df_testt2=pd.DataFrame(yp2)
df_testt2["iter"]=pd.DataFrame(df_col)
df_testt2.columns = ["value","iter"]
columnsTitles=["iter","value"]
df_testt2=df_testt2.reindex(columns=columnsTitles)
#Thrid and fourth reference set with 300 and combes samle four
df_testt2.to_csv('C:/python/combes reference set_distrib_xiii.csv', index=False,header=True)
dfz22.to_csv('C:/python/combes reference set_distrib_xiv.csv', index=False,header=True)
y2 = np.concatenate([np.array(i) for i in piittas]) #does the flattening
y2=y2[0:1000]
plt.plot(y2)
#https://pandas.pydata.org/pandas-docs/stable/reshaping.html
df_col=list(range(0,1001))
df_testt2=[]
df_testt2=pd.DataFrame(y2)
df_testt2["iter"]=pd.DataFrame(df_col)
df_testt2.columns = ["value","iter"]
columnsTitles=["iter","value"]
df_testt2=df_testt2.reindex(columns=columnsTitles)
df_testt2.to_csv('C:/python/combes reference set_distrib_x2.csv', index=False,header=True)
#https://stackoverflow.com/questions/30112202/how-do-i-find-the-closest-values-in-a-pandas-series-to-an-input-number
#http://pandas.pydata.org/pandas-docs/stable/merging.html
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.pivot.html
index=0
iitta=[]
for index in range(len(iitt)):
    iitta.append(np.array(iitt[index].sample(n=22)))  
#https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.DataFrame.sample.html
#https://stackoverflow.com/questions/3337301/numpy-matrix-to-array
#https://stackoverflow.com/questions/10346336/list-of-lists-into-numpy-array
y = np.concatenate([np.array(i) for i in iitta]) #does the flattening
y=y[0:1000]
plt.plot(y)
#https://stackoverflow.com/questions/10346336/list-of-lists-into-numpy-array
df_col=list(range(0,1001))
df_testt2=[]
df_testt2=pd.DataFrame(y)
df_testt2["iter"]=pd.DataFrame(df_col)
df_testt2.columns = ["value","iter"]
columnsTitles=["iter","value"]
df_testt2=df_testt2.reindex(columns=columnsTitles)
#https://stackoverflow.com/questions/25649429/how-to-swap-two-dataframe-columns
df_testt2.to_csv('C:/python/combes reference set_distrib3.csv', header=True)
my_df = pd.DataFrame(mmf2)
my_df.to_csv('exp-okish-tikka29918.csv', index=False, header=False)
displ_dfce2=dfce2_v3.ix[:,0]/dfce2_v3.ix[:,1]
dfce2_v3['µm/s']=displ_dfce2
dfce2_v4=dfce2_v3.set_index("sample")
ind2=np.array(list(dfce2_v3.ix[:,2].unique()))
dfce2_v3.ix[:,3].unique()
index=0
mmf2=[]
for index in ind2:
    mmf2.append(np.matrix(dfce2_v4.ix[index,2:4]))         
for i in range(len(mmf2)):
    mmf2[i][np.where(np.isnan(mmf2[i]))]=+0.00001
dfce2_v3.ix[:,3:6].pivot(index=None, columns='time', values='µm/s')
dfx = pd.DataFrame(mmf2[0])   
dfx.columns = ['time','µm/s']
dfx=dfx.set_index("time")
#remove zero values
dfx=dfx.loc[dfx['µm/s'] != 0.00001]
#https://pandas.pydata.org/pandas-docs/stable/reshaping.html
dfx2=dfce2_v3.ix[:,3:6].pivot(index=None, columns='time', values='µm/s')
dfx2_l=np.mean(dfx2)
dfx2_l.iloc[0]
dfx2_l2=dfx2_l.ix[2:73]
df_testt=dfx2_l2[dfx2_l2.values.repeat(15, axis=0)]
df_testt=pd.DataFrame(df_testt.index)
df_testt=df_testt[0:1000]
df_col=list(range(0,1001))
df_testt2=pd.DataFrame(df_testt)
df_testt2["value"]=df_testt
df_testt2.columns = ["iter","value"]
df_testt2["iter"]=pd.DataFrame(df_col)
#Fifth and sixth reference sets..
df_testt2.to_csv('C:/python/combes reference set_distrib.csv', header=True)
dfx2_l.to_csv('C:/python/a reference set2.csv', header=True)
for i in range(len(dfx)):
    Ashape=dfx.shape
    new_shape=len(dfx.index.unique())
    shape_diff = np.asarray(new_shape) - np.asarray(Ashape)
    dfx2[i,:]=np.lib.pad(dfx, (0,shape_diff), 'constant', constant_values=(0.1)) 
index=0
mmf3=[]
for index in ind2:
    mmf3.append(np.matrix(dfce2_v4.ix[index,2:4])) 
mmf2[7]=np.insert(mmf2[7],0,(24,0.00001), axis=0) #mmf2 in rollinw window function

#%% 5) ANALYSING NEW EXPERIMENTAL DATA, KIDNEY ORGANOID
df3_5b3d_sok=df3_5b3d_speed/60
df3_5b3d_sok=df3_5b3d_sok.T
df3_5b3d_sok=df3_5b3d_sok.fillna(0)
np.mean(df3_5b3d_sok)
np.mean(np.mean(df3_5b3d_sok))
speed_3d7b =np.mean(np.mean(df3_3d7b_speed))
df3_3d7b_speed.to_csv('speed7bok.csv',na_rep=0, header=None) 
len(remove_values_from_list(df3_5b3d_sok.ix[:,0], 0))  
np.min(remove_values_from_list(df3_5b3d_sok.ix[:,0], 0))#0.012448800000000001
np.std(remove_values_from_list(df3_5b3d_sok.ix[:,0], 0))#0.5945432640927301
np.mean(remove_values_from_list(df3_5b3d_sok.ix[:,0], 0))#0.669214758606375
df3_5b3d_sok_mean=np.mean(df3_5b3d_sok)
df3_5b3d_sok_mean=df3_5b3d_sok_mean[np.isfinite(df3_5b3d_sok_mean.ix[:])]
speed_5b3d =np.mean(np.mean(df3_5b3d_sok))#speed_5b3d_s=speed_5b3d*0.24/60, Out[10]: 0.0107 #ok!!
a=df3_5b3d_sok.ix[:,0]<1.26376
b=df3_5b3d_sok.ix[:,0]>0.01245
c=a & b
df3_5b3d_sok.ix[c,0]
axan=[]
npn=[]
nm=[]   
a=[]
b=[]
c=[]
for i in range(df3_5b3d_sok.shape[1]-1):
    npn.append(
            mean(remove_values_from_list(df3_5b3d_sok.ix[:,i],0))+
            std(remove_values_from_list(df3_5b3d_sok.ix[:,i],0)))
    a.append(df3_5b3d_sok.ix[:,i]<npn[i])
    b.append(df3_5b3d_sok.ix[:,i]>0.0125)
    c.append(a[i] & b[i])
    axan.append(df3_5b3d_sok.ix[c[i],i])
    bxan=[]
    for i in range(12):
        bxan.append((axan[i].shape))  
    dxan=[]
    for i in range(len(axan)):
        dxan.append(std(axan[i]))
        cxan=np.array(bxan)
hhn2 = [i * (92/max(bxana)) for i in bxana]
hhn2=hhn2[::-1]
#https://stackoverflow.com/questions/35166633/how-do-i-multiply-each-element-in-a-list-by-a-number
#https://stackoverflow.com/questions/10715965/add-one-row-to-pandas-dataframe
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
hhn4=np.array(hhn2).repeat(9, axis=0)
yes3=pd.DataFrame(hhn4)   
yes4=np.array(yes3[::-1])
yes4=np.array(yes4).repeat(2, axis=0)
yes4=pd.DataFrame(yes4)
yes4=yes4[0:130]
plt.plot(yes4)
yes4=pd.DataFrame(yes4)
yes4.to_csv('C:\python\yes5',index=False,header='infer') #Tallennus samassa  
#https://stackoverflow.com/questions/15868512/list-to-array-conversion-to-use-ravel-function

#%% 6) MAKING EXPERIMENTAL FRAMES READY FOR ROLLING MEAN CALCULATIONS
#https://stackoverflow.com/questions/38884466/how-to-select-a-range-of-values-in-a-pandas-dataframe-column
dfce2=dfc_experim2[['sample','tipendDist','time','speed','x','y','z']] 
dfce2=dfce2.dropna(axis=0) #note, axis=0 
d_som=dfc_experim2[['speed','tipendDist','time','x','y','z']] 
d_som=d_som.fillna(0)
d_som['speed']=d_som['speed']*60+max(d_som['speed']*60)/200
d_som['speed2']=d_som['speed']*2000
d_som['speed2']=d_som['speed2']/max(d_som['speed2'])+0.005
for i in range(len(d_som['speed2'])):
    if d_som['speed2'][i] >= 0.95:
        d_som['speed2'][i]=0.95
norm = mpl.colors.Normalize(vmin=-20, vmax=10)
cmap = cm.hot
x = 0.3
m = cm.ScalarMappable(norm=norm, cmap=cmap)
print(m.to_rgba(x))        
d_som.to_csv('C:/python/irist.csv', index=False,header=False)
squares = np.unique(dfce2['sample'])
dfce22=dfce2.loc[dfce2['sample'].isin(squares)]
lot=(tuple(range(3,50)))
#https://cmdlinetips.com/2018/02/how-to-subset-pandas-dataframe-based-on-values-of-a-column/
dfz20=dfce22[dfce22.time.isin(lot)] 
dfz20=dfz20.set_index("time") 
okoo=[]
okoo2=[]
okx=[]
oky=[]
okz=[]
index=0
loti=np.unique(dfz20.index)
for index in loti:
    okoo.append(dfz20.ix[index,1]) 
    okoo2.append(dfz20.ix[index,2]) 
    okx.append(dfz20.ix[index,3]) 
    oky.append(dfz20.ix[index,4])     
    okz.append(dfz20.ix[index,5])    
b = np.zeros([len(okoo),len(max(okoo,key = lambda x: len(x)))])
b1 = np.zeros([len(okoo2),len(max(okoo2,key = lambda x: len(x)))])
b2 = np.zeros([len(okx),len(max(okx,key = lambda x: len(x)))])
b3 = np.zeros([len(oky),len(max(oky,key = lambda x: len(x)))])
b4 = np.zeros([len(okz),len(max(okz,key = lambda x: len(x)))])
for i,j in enumerate(okoo):
    b[i][0:len(j)] = j #toimi, viimeinkin!!
m = np.mean(b[b > 0])
b[b == 0] = m
b=b.T 
for i,j in enumerate(okoo2):
    b1[i][0:len(j)] = j #toimi, viimeinkin!!
m = np.mean(b1[b1 > 0])
b1[b1 == 0] = m
b1=b1.T 
for i,j in enumerate(okx):
    b2[i][0:len(j)] = j #toimi, viimeinkin!!
m = np.mean(b2[b2 > 0])
b2[b2 == 0] = m
b2=b2.T 
for i,j in enumerate(oky):
    b3[i][0:len(j)] = j 
m = np.mean(b3[b3 > 0])
b3[b3 == 0] = m
b3=b3.T
for i,j in enumerate(okz):
    b4[i][0:len(j)] = j #
m = np.mean(b4[b4 > 0])
b4[b4 == 0] = m
b4=b4.T
len(np.mean(b, axis=1)) #across cells is better than time: plt.plot(np.mean(b, axis=0))
plt.title("All Cells")
plt.xlabel("cells")
plt.ylabel("speed")
plt.plot(np.mean(b1, axis=1)*60)

#%% 7) ROLLING WINDOW FOR THE NEW EXPERIMENTAL DATA, KIDNEY ORGANOID      
#The experimental values to b matrix are obtained above for the rollwing window
zz=rolling_window(b1[80:180]*60, 3) #103X48 matrice, i.e 103 data measurments per whole 48 time space
yy=np.mean(zz, -1) 
yz=np.mean(yy, 0) 
plt.figure()
plt.plot(yz)
okok=[]
for i in range(0,(len(yz)-1)):
    okok.append(np.arange(yz[i], yz[i+1], (yz[i+1] - yz[i])/(1000/len(yz))))
#Finally the calculation 
okok2 = np.concatenate([np.array(i) for i in okok])
okok2=okok2[0:1001] 
okok2=np.random.choice(np.concatenate([np.array(i) for i in b1*60]), 1001, False)   
df_col=list(range(0,1001))
plt.figure()
plt.xlabel('Time (MCS)', fontsize=18)
plt.ylabel('Speed (μm/min)', fontsize=18)
plt.plot(okok2) #fig.savefig('Rolling mean to Combes data_tip.jpg')
fig.savefig('Rolling mean to Combes data.jpg')
df_testt2=[]
df_testt2=pd.DataFrame(okok2)
df_testt2["iter"]=pd.DataFrame(df_col)
df_testt2.columns = ["value","iter"]
columnsTitles=["iter","value"]
df_testt2=df_testt2.reindex(columns=columnsTitles)
df_testt2.to_csv('C:/pyintro/moving average speed set from Combes.csv', index=False,header=True)
