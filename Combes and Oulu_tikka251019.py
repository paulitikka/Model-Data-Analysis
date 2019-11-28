#%% # EXPERIMENTAL FILES 
# Combes et al. 2018, An excel file:
dfc_experim = pd.read_excel('C:/python/' + 'mmc5.xlsx')
dfc_experim2 = pd.read_excel('C:/python/' + 'mmc4.xlsx')
# Saarela et al. 2017, these are the nephron progenitor cell speeds of the Oulu experiments (2016/17)
df1_experimi = pd.read_excel('C:/python/' + 'right-speed.xls', header=None)
df2_experimi = pd.read_excel('C:/python/' + 'left-speed.xls', header=None)
df3_experimi = pd.read_excel('C:/python/' + 'Whole UB tip-speed.xls', header=None)
# 3/20180 Oulu tests tests(?)
#df3_ee = pd.read_csv('C:/python/speed.csv', delimiter=',', header=None)

#%%
#you may also use all the samples
dfce2_v3=dfc_experim2[['sample','time','speed','y']]

#%%
dfce2_v3=dfce2_v3[95:]
#dfce2_v3[np.isnan(dfce2_v3)]=0
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
import numpy as np
import scipy.stats
def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    
        return m, m-h, m+h    
yes=[]
yes2=[]
yes3=[]
for i in range(time):
    yes.append(mean_confidence_interval(result[[i]], confidence=0.95))


dfz=dfce2_v3.set_index("sample")
yn = [(float(i)-min(dfz['y']))/(max(dfz['y'])-min(dfz['y'])) for i in dfz['y']]
dfz['yn']=yn
dfz2yn=dfz[dfz['yn']>0.3818340295]
        
#%% iteroidaan x (0.1-0.5): Okt: 0.1,0.2,0.3->0.4 ei ok 
#->0.35->0.37...->0.381834->0.3818340295
        #target above: 4, 5, 6, 11, 13, 14, 16, 17
len(dfz[dfz['yn']>0.3818340295].index.unique()) #tulis olla 8

#%%
dfz2yn=pd.DataFrame(dfz2yn)
index = 0
ind=[4, 6, 11, 13, 14, 16, 17]
ok=[]

for index in ind:
    ok.append(dfz2yn['time'].ix[index].value_counts())
    #%%
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
#%%
matrix = []
for i in range(len(hh2)):
    matrix.append([hh2[i], hh2[i].index])
    #%%
hh3=np.array(hh2) #important
#    https://cognitiveclass.ai/blog/nested-lists-multidimensional-numpy-arrays
hh4=hh3.T
        #%
yes=[]
for i in range(len(hh4)):
    yes.append(mean_confidence_interval(hh4[i], confidence=0.95))        
   #%                              
yes=pd.DataFrame(yes)
#    yes2=pd.DataFrame(yes2) 
#    yes3=pd.DataFrame(yes3) 
#%    
for i in range(len(yes)):
    yes.loc[i,1]=float(yes.loc[i,1])
    yes.loc[i,2]=float(yes.loc[i,2])

yes[1] = yes[1].astype(float)
yes[2] = yes[2].astype(float) 
#%%
yes2=np.repeat(yes, 2)
#%%
yes2=pd.DataFrame()
yes2[0]=np.repeat(yes.loc[:,0], 2)
yes2[1]=np.repeat(yes.loc[:,1], 2)
yes2[2]=np.repeat(yes.loc[:,2], 2)
jaa=yes[::-1]    
jaa2=list(jaa[0])
jaa2=np.repeat(jaa2, 2)    
yes3=pd.DataFrame(jaa2)   
#%%

yes3.to_csv('C:\python\yes2',index=False,header='infer') #Tallennus samassa     
#%%
yes3=pd.DataFrame(hhn2)  
yes3.to_csv('C:\python\yesi',index=False,header='infer') 

#%%
#https://stackoverflow.com/questions/24225072/repeating-elements-of-a-list-n-times/24225106  
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html
#https://cognitiveclass.ai/blog/nested-lists-multidimensional-numpy-arrays
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_index.html
#https://stackoverflow.com/questions/6422700/how-to-get-indices-of-a-sorted-array-in-python
#https://stackoverflow.com/questions/38309729/count-unique-values-with-pandas-per-groups/38309823                           
                                     
#%% 
ok=dfz2yn['time'].value_counts()
heihou=ok.sort_index()
hh2=heihou*(150/np.max(heihou))       
        



#%%
dfz2=dfz2.set_index("time")
#%%
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
df_testt2.to_csv('C:/python/a reference set_speed_test.csv', header=True, index=None )
dfz22=dfz2['speed']
index=0
iittas=[]
piittas=[]
for index in list(lot):
    iittas.append(np.array(dfz22[index].sample(n=22)))
    #%%
piittas=[]    
for index in list(lot):    
    piittas.append(dfz22[index].iloc[(dfz22[index]-np.mean(dfz22[index])).abs().argsort()[:22]])

ypp = np.concatenate([np.array(i) for i in piittas])
ypp=ypp[0:1000]
    #%%
sort(dfz22[49].iloc[(dfz22[49]-np.mean(dfz22[49])).abs().argsort()[:22]])
#%% Combes tipEndDistances
#https://stackoverflow.com/questions/10346336/list-of-lists-into-numpy-array
dfce2_v3=dfc_experim2[['sample','time','tipendDist']] 
dfce2_v3=dfce2_v3.dropna(axis=0) #note, axis=0  
dfz2=dfce2_v3[dfce2_v3['sample']==4] #riittää määrää
#%
dfz2=dfz2.set_index("time")
dfz22=dfz2['tipendDist']
combes=np.random.choice(dfz22, 3500, False)

#%%
import random
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
df_testt2.to_csv('C:/python/a reference set_ub3d_adh2_tot_ok.csv', header=True, index=None )
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html
#%%
lot=list(range(2,50))
index=0
iittas=[]
for index in list(lot):
    iittas.append(np.array(dfz22[index].sample(n=22)))
    #%
yp = np.concatenate([np.array(i) for i in iittas]) #does the flattening
yp=yp[0:4999]
plt.plot(yp)
#%%
yp2=ypp[0:300]
#%
#https://pandas.pydata.org/pandas-docs/stable/reshaping.html
df_col=list(range(0,301))
df_testt2=[]
df_testt2=pd.DataFrame(yp2)
df_testt2["iter"]=pd.DataFrame(df_col)
df_testt2.columns = ["value","iter"]
columnsTitles=["iter","value"]
df_testt2=df_testt2.reindex(columns=columnsTitles)
df_testt2.to_csv('C:/python/combes reference set_distrib_xiii.csv', index=False,header=True)
#%%
dfz22.to_csv('C:/python/combes reference set_distrib_xiv.csv', index=False,header=True)

#%%
y2 = np.concatenate([np.array(i) for i in piittas]) #does the flattening
y2=y2[0:1000]
plt.plot(y2)
#%%
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

#%
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
#%%
my_df.to_csv('exp-okish-tikka29918.csv', index=False, header=False)
#%%
displ_dfce2=dfce2_v3.ix[:,0]/dfce2_v3.ix[:,1]
dfce2_v3['µm/s']=displ_dfce2

dfce2_v4=dfce2_v3.set_index("sample")

ind2=np.array(list(dfce2_v3.ix[:,2].unique()))
dfce2_v3.ix[:,3].unique()
#dfce2_v5=dfce2_v4.ix[:,[3,2]]
#dfce2_v5.columns = ['µm/s', 'time']
index=0
mmf2=[]
for index in ind2:
    mmf2.append(np.matrix(dfce2_v4.ix[index,2:4]))  

#for i in range(len(mmf2)):
#        mmf2[i][np.where(np.isnan(mmf2[i]))]=+0.00001
#        
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
#%%
dfx2_l2=dfx2_l.ix[2:73]
#dfx2_l2.DataFrame(dfx2_l2.values.repeat(10, axis=0), columns=dfx2_l2.columns)

df_testt=dfx2_l2[dfx2_l2.values.repeat(15, axis=0)]
df_testt=pd.DataFrame(df_testt.index)
df_testt=df_testt[0:1000]
#%%
df_col=list(range(0,1001))

df_testt2=pd.DataFrame(df_testt)
df_testt2["value"]=df_testt
#%%
#dfxz = pd.DataFrame()   
df_testt2.columns = ["iter","value"]
df_testt2["iter"]=pd.DataFrame(df_col)

df_testt2.to_csv('C:/python/combes reference set_distrib.csv', header=True)

dfx2_l.to_csv('C:/python/a reference set2.csv', header=True)

#%%
for i in range(len(dfx)):
    Ashape=dfx.shape
    new_shape=len(dfx.index.unique())
    shape_diff = np.asarray(new_shape) - np.asarray(Ashape)
    dfx2[i,:]=np.lib.pad(dfx, (0,shape_diff), 'constant', constant_values=(0.1)) 


#%%
index=0
mmf3=[]
for index in ind2:
    mmf3.append(np.matrix(dfce2_v4.ix[index,2:4])) 
    
    #%%but
#mmf2[7]=np.delete(mmf2[7],axis=1)    
mmf2[7]=np.insert(mmf2[7],0,(24,0.00001), axis=0)
#%% 3/2018 oulu experiment recut from 7b:
df3_3d7b_speed = pd.read_csv('C:/python/speed_7b3d.csv', delimiter=',', header=None)
speed_3d7b =np.mean(np.mean(df3_3d7b_speed))
#speed_3d7b_s=speed_3d7b*0.24/60 #0.24 was the conversion -> 0.010148787434750365, which is close to my earlier resutls!

#%%
df3_3d7b_speed.to_csv('speed7bok.csv',na_rep=0, header=None) 
#%%
#% 3/2018 oulu experiment recut from 5b, also 0.24
df3_5b3d_speed = pd.read_csv('C:/python/speed_5b3d.csv', delimiter=',', header=None)
#%%
df3_5b3d_sok=df3_5b3d_speed/60
df3_5b3d_sok=df3_5b3d_sok.T
df3_5b3d_sok=df3_5b3d_sok.fillna(0)

np.mean(df3_5b3d_sok)
np.mean(np.mean(df3_5b3d_sok))
#%%
def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]
#%%
len(remove_values_from_list(df3_5b3d_sok.ix[:,0], 0))  
np.min(remove_values_from_list(df3_5b3d_sok.ix[:,0], 0))#0.012448800000000001
np.std(remove_values_from_list(df3_5b3d_sok.ix[:,0], 0))#0.5945432640927301
np.mean(remove_values_from_list(df3_5b3d_sok.ix[:,0], 0))#0.669214758606375
#%%
0.5945432640927301+0.669214758606375 #1.263758022699105
#%%
df3_5b3d_sok_mean=np.mean(df3_5b3d_sok)
df3_5b3d_sok_mean=df3_5b3d_sok_mean[np.isfinite(df3_5b3d_sok_mean.ix[:])]
speed_5b3d =np.mean(np.mean(df3_5b3d_sok))
#speed_5b3d_s=speed_5b3d*0.24/60
#speed_5b3d_s : Out[10]: 0.010668674407168734 #ok!!
#%%

a=df3_5b3d_sok.ix[:,0]<1.26376
b=df3_5b3d_sok.ix[:,0]>0.01245
c=a & b
#%%
df3_5b3d_sok.ix[c,0]
#%%
3316/4
#%%
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
   #%
#    nm.append(
#            min(remove_values_from_list(df3_5b3d_sok.ix[:,i],0)))
    #%
    a.append(df3_5b3d_sok.ix[:,i]<npn[i])
    #%
    b.append(df3_5b3d_sok.ix[:,i]>0.0125)
    c.append(a[i] & b[i])

    axan.append(df3_5b3d_sok.ix[c[i],i])
    
    #%%
    bxan=[]
    for i in range(12):
        bxan.append((axan[i].shape))
       #%%
       
    dxan=[]
    for i in range(len(axan)):
        dxan.append(std(axan[i]))
        #%%
#        std(bxan)
#Out[81]: 199.40550532910459
        cxan=np.array(bxan)
        #%%
def convertTuple(tup): 
    str =  ''.join(tup) 
    return str
#%%
    bxan=ravel(bxan)
    #%
#    bxan=list(bxan[::-1])
    #%%
#    bxan[0]= 3490+199  #array([3323, 3657,+std
    bxan=np.delete(bxan,obj=1)
    #%%
    hhn=
    #%%
    bxana=[]
    for i in range(len(bxan)):
        bxana.append(bxan[i]-2*(bxan[i]-3689))
    #%%
#hh2=[]
#for i in range(len(bxan)):    
#    #%
#hhn=bxan*(103/min(bxan)) 
hhn2 = [i * (92/max(bxana)) for i in bxana]
#%%
hhn2=hhn2[::-1]

#https://stackoverflow.com/questions/35166633/how-do-i-multiply-each-element-in-a-list-by-a-number
#https://stackoverflow.com/questions/10715965/add-one-row-to-pandas-dataframe
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
hhn4=np.array(hhn2).repeat(9, axis=0)
yes3=pd.DataFrame(hhn4)   
#%%
yes4=np.array(yes3[::-1])
#%%
#yes4=yes3+5
#%%
yes4=np.array(yes4).repeat(2, axis=0)

yes4=pd.DataFrame(yes4)
yes4=yes4[0:130]
plt.plot(yes4)
#%%
yes4=pd.DataFrame(yes4)
yes4.to_csv('C:\python\yes5',index=False,header='infer') #Tallennus samassa  
#    199.40550532910459/mean(bxan)
#Out[108]: 0.05115147635633294 #std is around 5%
#    https://stackoverflow.com/questions/15868512/list-to-array-conversion-to-use-ravel-function
#%%Experimental frames
#https://stackoverflow.com/questions/38884466/how-to-select-a-range-of-values-in-a-pandas-dataframe-column
dfce2=dfc_experim2[['sample','tipendDist','time','speed','x','y','z']] 
dfce2=dfce2.dropna(axis=0) #note, axis=0 
#%%
d_som=dfc_experim2[['speed','tipendDist','time','x','y','z']] 
d_som=d_som.fillna(0)
d_som['speed']=d_som['speed']*60+max(d_som['speed']*60)/200
d_som['speed2']=d_som['speed']*2000
d_som['speed2']=d_som['speed2']/max(d_som['speed2'])+0.005

for i in range(len(d_som['speed2'])):
    if d_som['speed2'][i] >= 0.95:
        d_som['speed2'][i]=0.95
import matplotlib as mpl
import matplotlib.cm as cm
norm = mpl.colors.Normalize(vmin=-20, vmax=10)
cmap = cm.hot
x = 0.3
m = cm.ScalarMappable(norm=norm, cmap=cmap)
print(m.to_rgba(x))        
d_som.to_csv('C:/python/irist.csv', index=False,header=False)
#%%
squares = np.unique(dfce2['sample'])
dfce22=dfce2.loc[dfce2['sample'].isin(squares)]
#%
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
    
#%  
b = np.zeros([len(okoo),len(max(okoo,key = lambda x: len(x)))])
b1 = np.zeros([len(okoo2),len(max(okoo2,key = lambda x: len(x)))])
b2 = np.zeros([len(okx),len(max(okx,key = lambda x: len(x)))])
b3 = np.zeros([len(oky),len(max(oky,key = lambda x: len(x)))])
b4 = np.zeros([len(okz),len(max(okz,key = lambda x: len(x)))])
#%
for i,j in enumerate(okoo):
    b[i][0:len(j)] = j #toimi, viimeinkin!!
m = np.mean(b[b > 0])
b[b == 0] = m
b=b.T 
#%
for i,j in enumerate(okoo2):
    b1[i][0:len(j)] = j #toimi, viimeinkin!!
m = np.mean(b1[b1 > 0])
b1[b1 == 0] = m
b1=b1.T 

#%
for i,j in enumerate(okx):
    b2[i][0:len(j)] = j #toimi, viimeinkin!!
m = np.mean(b2[b2 > 0])
b2[b2 == 0] = m
b2=b2.T 

#%
for i,j in enumerate(oky):
    b3[i][0:len(j)] = j #toimi, viimeinkin!!
m = np.mean(b3[b3 > 0])
b3[b3 == 0] = m
b3=b3.T
#%
for i,j in enumerate(okz):
    b4[i][0:len(j)] = j #toimi, viimeinkin!!
m = np.mean(b4[b4 > 0])
b4[b4 == 0] = m
b4=b4.T

#% 
#plt.plot(np.mean(b, axis=0))  #accross time
len(np.mean(b, axis=1)) #across cells is better
plt.title("All Cells")
plt.xlabel("cells")
plt.ylabel("speed")
plt.plot(np.mean(b1, axis=1)*60)

bok2=np.mean(b, axis=1)[325:425] #for tip
bok2i=np.mean(b, axis=1)[180:280] #for conrer
bokzu=np.mean(b1, axis=1)[80:180]*60 #for rnd

tip_speed=np.mean(b, axis=1)[325:425]
corner_speed=np.mean(b, axis=1)[180:280]

speed_all=np.mean(b, axis=1)

#%
bx=np.mean(b2, axis=1)
by=np.mean(b3, axis=1)
bz=np.mean(b4, axis=1)


bx_t=bx[325:425] #for tip
bx_c=bx[180:280] #for conrer

#%%
plt.plot(bx_t)
plt.plot(bx_c)

#%%
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(bx, by)


#%%
N = len(bx)
x = bx/max(bx)
y = by/max(by)
colors = np.random.rand(N)
area = (30 * by/max(by))**2   # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()

#%%Tip
N = len(bx[325:425])
xe = bx[325:425]/max(bx[325:425])
ye = by[325:425]/max(by[325:425])
ze=bz[325:425]/max(bz[325:425])
#colors = np.random.rand(N)
colors = tip_speed/max(tip_speed) 
colors=np.round(colors,1)
area = (30 * ze)**2   # 0 to 15 point radii
#%%tip
N = len(bx[325:425])
x = bx[325:425]
y = by[325:425]
z=bz[325:425]
colors = np.random.rand(N)
area = (30 * z)**2   # 0 to 15 point radii

#%%
def corr2hex(n):
    ''' Maps a number in [0, 1] to a hex string '''
    if n == 1: return '#fffff'
    else: return '#' + hex(int(n * 16**6))[2:].zfill(6)

print(corr2hex(0.31))

#%% Good, 19.11.18
import matplotlib.cm as cm
oki=[]
for i in range(len(speed_colors)):
    oki.append(corr2hex(speed_colors[i]))

#%%Corner
N = len(bx[180:280])
x = bx[180:280]/max(bx[180:280])
y = by[180:280]/max(by[180:280])
z=bz[180:280]/max(bz[180:280])
colors = corner_speed/max(corner_speed) 
colors=np.round(colors,1) 
#oki
area = (30 * z)**2   # 0 to 15 point radii

#%%corner
import matplotlib.colors as clr

N = len(bx[180:280])
x = bx[180:280]
y = by[180:280]
z=bz[180:280]
speed_colors=np.array(corner_speed/max(corner_speed))
speed_clr=[]
for i in range(len(speed_colors)):
    speed_clr.append((corner_speed/max(corner_speed))[i])
colors = clr.hsv_to_rgb(speed_clr) #np.random.rand(N)
area = (300 * z)**2   # 0 to 15 point radii

#%%
import matplotlib.cm as cm

#%%Absolutely all Combes data
N = len(bx)
x = bx/max(bx)
y = by/max(by)
z = bz/max(bz)
colors = speed_all/max(speed_all) 
#colors=np.round(colors,4)

for i in range(len(y)):
    if x[i] >= 0.95:
        x[i]=0.95
    if x[i] <= 0.05:
        x[i]=0.05
    if y[i] >= 0.95:
        y[i]=0.95
    if y[i] <= 0.05:
        y[i]=0.05    
    if z[i] >= 0.95:
        z[i]=0.95
    if z[i] <= 0.05:
        z[i]=0.05 
    if colors[i] >= 0.95:
        colors[i]=0.95
    if colors[i] <= 0.01:
        colors[i]=0.1

colors2=colors*200
t=np.array(tuple(range(len(colors2))))
ta=np.round(colors2,0)

for i in range(len(colors2)):
    t[i]=int(ta[i])-int(min(ta)) 
#oki
area = (30 * z)**2   # 0 to 15 point radii
#%%
x = [(float(i)-min(x))/(max(x)-min(x)) for i in x]
x=np.round(x,2)+1.5
y = [(float(i)-min(y))/(max(y)-min(y)) for i in y]
y=np.round(y,2)+1.5
z = [(float(i)-min(z))/(max(z)-min(z)) for i in z]
z=np.round(z,2)+1.5
colors = [(float(i)-min(colors))/(max(colors)-min(colors)) for i in colors]
colors=np.round(colors,2)+1.5
colors2=colors*10


#%%
df_testt=pd.DataFrame(x)
df_testt["y"]=pd.DataFrame(y)
df_testt["z"]=pd.DataFrame(z)
df_testt["colors"]=pd.DataFrame(colors)
df_testt["t"]=pd.DataFrame(colors)
df_testt.to_csv('C:/python/iriste.csv', index=False,header=False)

#%%Oulu Data
# Oulu 2018 positions->distances, always check the header
df_oulu_x = pd.read_csv('C:/python/position_1_7b3d.csv', delimiter=',', header=None)
df_oulu_y = pd.read_csv('C:/python/position_2_7b3d.csv', delimiter=',', header=None)
df_oulu_z = pd.read_csv('C:/python/position_3_7b3d.csv', delimiter=',', header=None)
#%speed
df_speed_ilya7b = pd.read_csv('C:/python/speed_7b3d.csv', delimiter=',', header=None)

#2.5 is higher then zero, and variance is great (25) but nans still can be 0 for calc
#%%
x=np.mean(df_oulu_x, axis=1)
y=np.mean(df_oulu_y, axis=1)
z=np.mean(df_oulu_z, axis=1)
colors=np.mean(df_speed_ilya7b, axis=1)

#%%
x[np.isnan(x)]=round(min(x),2)+np.int(var(x)*np.random.rand(1)*0.1) #slightly more than min value of -0.08, towards max value,0.06
y[np.isnan(y)]=round(min(y),2)+np.int(var(y)*np.random.rand(1)*0.1) #similarly
z[np.isnan(z)]=round(min(z),2)+np.int(var(z)*np.random.rand(1)*0.1) #similarly
colors[np.isnan(colors)]=round(min(colors),2)+np.int(var(colors)*np.random.rand(1)*0.1) #similarly

#%%



x = [(float(i)-min(x))/(max(x)-min(x)) for i in x]
x=np.round(x,2)+1.5

y = [(float(i)-min(y))/(max(y)-min(y)) for i in y]
y=np.round(y,2)+1.5

z = [(float(i)-min(z))/(max(z)-min(z)) for i in z]
z=np.round(z,2)+1.5

colors = [(float(i)-min(colors))/(max(colors)-min(colors)) for i in colors]
colors=np.round(colors,2)+1.5

colors2=colors*10
area = (30 * z)**2
#%%
df_testt=pd.DataFrame(x)
df_testt2=pd.DataFrame(y)
df_testt3=pd.DataFrame(z)
df_testt4=pd.DataFrame(colors)
df_testt5=pd.DataFrame(colors2)
result = pd.concat([df_testt, df_testt2,df_testt3, df_testt4,df_testt5], axis=1, sort=False)
#%%
result.to_csv('C:/python/irisn.csv', index=False,header=False)
#z=test2[]
b = np.zeros([len(speeds),len(max(speeds,key = lambda x: len(x)))])
#xyza = np.zeros([len(xyzz),len(max(xyzz,key = lambda x: len(x)))])


#%
for i,j in enumerate(speeds):
    b[i][0:len(j)] = j #toimi, viimeinkin!!
    #%%
m = np.mean(b[b > 0])
b[b == 0] = m
b=b.T 

#%%
#    speed = np.concatenate([np.array(i) for i in speeds]) 
#speed = np.mean(speeds, axis=0)
#xyzi=np.mean(xyzz, axis=1)
def model_to_semi_som(df_pos_np3d):
    time=len((df_pos_np3d.set_index("time_mcs")).index.unique())
    dfoki2=[]
    dfi =[]
    #indeces=[]
    dfoki2=df_pos_np3d.set_index("time_mcs").set_index("cell_no")
    #indeces=np.array(dfoki2.index.unique())
    criteria=dfcell_names3da[1]=='NPCells'
    listi=dfcell_names3da[0][criteria]
    dfooo=dfoki2.ix[np.array(listi)]
    indeces=np.array(listi)
    index=0
    mmf=[]
    mmf2=[]
    for index in indeces:
        mmf.append(np.array(dfooo.ix[index,0:3]))  #NP all coordinates
    #%
    time=2000 
    cells=392 #
    inst_speed_np3d=inst_fun(df_pos_np3d,time,cells) 
    #%
    criteria2=dfcell_names3da[1]!='Wall'
    listia=dfcell_names3da[0][criteria2]
    len(listia)
    listia=pd.DataFrame(listia)
    listia.index = range(392)
    #    https://stackoverflow.com/questions/19609631/python-changing-row-index-of-pandas-data-frame
    #https://www.geeksforgeeks.org/python-intersection-two-lists/
    #%
    listia.loc[listia[0].isin(listi)] 
    npt=listia.loc[listia[0].isin(listi)].index
    
    speed_np3d=inst_speed_np3d[npt]
    speeds_np3d=np.mean(speed_np3d,axis=1)
    #https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
    #%
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


#%%
    x, y, z, colors, area =model_to_semi_som(df_pos_ub3d)

#%%continues
N = len(x)
x = abs(x/max(x))
y = abs(y/max(y))
z = abs(z/max(z))
colors = abs(colors/max(colors) )
T=abs(T/max(T))
for i in range(len(y)):
    if x[i] >= 0.95:
        x[i]=0.95
    if x[i] <= 0.05:
        x[i]=0.05
    if y[i] >= 0.95:
        y[i]=0.95
    if y[i] <= 0.05:
        y[i]=0.05    
    if z[i] >= 0.95:
        z[i]=0.95
    if z[i] <= 0.05:
        z[i]=0.05 
    if colors[i] >= 0.95:
        colors[i]=0.95
    if colors[i] <= 0.05:
        colors[i]=0.05
    if T[i] >= 0.95:
        T[i]=0.95
    if T[i] <= 0.05:
        T[i]=0.05
colors=np.round(colors,3) 
import numpy as np
import matplotlib.pyplot as plt
#https://matplotlib.org/users/pyplot_tutorial.html
plt.xlabel("X")
plt.ylabel("Y")
plt.axis([0, 1.1, 0, 0.55])
c_norm = [float(i)/max(colors) for i in colors] #replace your total color file
norm = [(float(i)-min(colors))/(max(colors)-min(colors)) for i in colors]
norm=np.round(norm,1) 
norm2=norm
#https://stackoverflow.com/questions/12487060/matplotlib-color-according-to-class-labels
#https://stackoverflow.com/questions/1540049/replace-values-in-list-using-python
for i in range(len(norm)):
    if norm[i] == 0.0:
        norm2[i]=0.05
    if norm[i] == 1.0:
        norm2[i]=0.95
plt.axis([0, 1.1, 0, 0.55])
for i in range(len(x)):
    plt.scatter(x[i], y[i],alpha=0.6, 
                s=area[i],c=cm.Spectral(norm2[i]))   #remember cm, and sometimes big letters
    
#https://matplotlib.org/users/colormaps.html     
#https://stackoverflow.com/questions/43121584/matplotlib-scatterplot-x-axis-labels
#   https://stackoverflow.com/questions/26785354/normalizing-a-list-of-numbers-in-python
#   https://matplotlib.org/gallery/color/colormap_reference.html
#https://stackoverflow.com/questions/8931268/using-colormaps-to-set-color-of-line-in-matplotlib

#%%Tot
N = len(bx[325:425])+len(bx[180:280])
x = bx[325:425]/max(bx[325:425])
y = by[325:425]/max(by[325:425])
z=bz[325:425]/max(bz[325:425])

x1 = bx[180:280]/max(bx[180:280])
y1 = by[180:280]/max(by[180:280])
z1=bz[180:280]/max(bz[180:280])

colors = corner_speed/max(corner_speed) 
colors=np.round(colors,1) 

colors1 =tip_speed/max(tip_speed) 
colors1=np.round(colors1,1)
colors=np.append(colors,colors1)


#np.random.rand(N) # 0 to 15 point radii

x2= [*x, *x1]
x2=np.array(x2)
y2= [*y, *y1]
zy=np.array(y2)
z2= [*z, *z1]
z2=np.array(z2)
area = (30 * z2)**2

#%%
N = len(bx[325:425])+len(bx[180:280])
x = bx[325:425]
y = by[325:425]
z=bz[325:425]

x1 = bx[180:280]
y1 = by[180:280]
z1=bz[180:280]

colors1 =tip_speed/max(tip_speed) 
colors1=np.round(colors1,1)
c_tot=np.append(colors,colors1)

#= np.random.rand(N) # 0 to 15 point radii

x2= [*x, *x1]
x2=np.array(x2)
y2= [*y, *y1]
zy=np.array(y2)
z2= [*z, *z1]
z2=np.array(z2)
area = (30 * z2)**2


#%%
N=100
m = np.random.randint(1,100, N)
cgy = np.sum(y*m)/np.sum(m) #tip:330.5437196368456, corner:170.6173017934623
cgx = np.sum(x*m)/np.sum(m) #tip:228.243323116723, corner:117.95584696056687
cgz = np.sum(z*m)/np.sum(m) #tip:16.60525099303062, corner:21.753953293621368

cgy1 = np.sum(y1*m)/np.sum(m) #tip:330.5437196368456, corner:170.6173017934623
cgx1 = np.sum(x1*m)/np.sum(m) #tip:228.243323116723, corner:117.95584696056687
cgz1 = np.sum(z1*m)/np.sum(m) #tip:16.60525099303062, corner:21.753953293621368


#%%
from mpl_toolkits.mplot3d import axes3d, Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#%
#plt.xlabel("X / µm")
#plt.ylabel("Y(normalized)")
#plt.zlabel("Y(normalized)")
ax = Axes3D(fig)
ax.scatter(x2, y2, z2,color=c_tot)
#ax.scatter(cgx, cgy, cgz,color='k', marker='+', s=1e4) #tip
#ax.scatter(cgx1, cgy1, cgz1,color='y', marker='+', s=1e4) #corner
plt.xlabel('x/µm')
plt.ylabel('y/µm')
plt.zlabel('z/µm')
#plt.scatter(x2, y2, s=area, c=colors, alpha=0.5)
plt.title('3 Dimensional Points, corner yellow');
plt.show()
#https://matplotlib.org/gallery/shapes_and_collections/scatter.htmlv
#https://stackoverflow.com/questions/1720421/how-to-concatenate-two-lists-in-python
#https://stackoverflow.com/questions/1614236/in-python-how-do-i-convert-all-of-the-items-in-a-list-to-floats
#https://www.digitalocean.com/community/tutorials/how-to-convert-data-types-in-python-3
#http://ifcuriousthenlearn.com/blog/2015/06/18/center-of-mass/
#https://matplotlib.org/2.1.1/gallery/mplot3d/scatter3d.html
#https://stackoverflow.com/questions/3810865/matplotlib-unknown-projection-3d-error



#%%
plt.title("Green is corner, Blue is tip")
plt.xlabel("Cell NO")
plt.ylabel("Speed micrometers/min")
plt.plot(speed_NP_corner,c='green')
plt.plot(speed_MM_tip)
#%%
speed_NP_corner = np.mean(b1, axis=1)[180:280]*60 
speed_MM_tip = np.mean(b1, axis=1)[325:425]*60 #tip ja corner väärinpäin...
#%%
speed_NP_RND = np.mean(b1, axis=1)[80:180]*60 
#np.random.choice(np.concatenate([np.array(i) for i in b1*60]), 1001, False)
#%%


#The experimental values to b matrix are obtained above for the rollwing window
zz=rolling_window(b1[80:180]*60, 3) #103X48 matrice, i.e 103 data measurments per whole 48 time space
yy=np.mean(zz, -1) 
yz=np.mean(yy, 0) 
plt.figure()
plt.plot(yz)
#%%
#
okok=[]
for i in range(0,(len(yz)-1)):
    okok.append(np.arange(yz[i], yz[i+1], (yz[i+1] - yz[i])/(1000/len(yz))))

#Finally the calculation 
okok2 = np.concatenate([np.array(i) for i in okok])
okok2=okok2[0:1001] 
#%%
okok2=np.random.choice(np.concatenate([np.array(i) for i in b1*60]), 1001, False)   
df_col=list(range(0,1001))
plt.figure()
plt.xlabel('Time (MCS)', fontsize=18)
plt.ylabel('Speed (μm/min)', fontsize=18)

plt.plot(okok2)
#fig.savefig('Rolling mean to Combes data_tip.jpg')
