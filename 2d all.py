# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:00:57 2019

@author: pauli
"""

# 2D ANALYSIS codes and routines, tikka 7.8.19, 8 phases:
# 1. import packages, 2. import model data, 3. peforming the calculations with functions,
# 4-7. create the functions for the calculations, and 8. create a good fucntion for data skimming 
# and importing from your folder systems

#%% 1) IMPORT PACKAGES
import pandas as pd #for importing files
# https://pandas.pydata.org/pandas-docs/version/0.18.1/generated/pandas.DataFrame.html
import numpy as np  #for calculations, array manipulations, and fun :)
import matplotlib.pyplot as plt #for scientifical plots
import random #for randomizing indeces
from random import randrange, uniform #just in case here, needed for random indexing    
import pylab #correct x and y limits after plotting
import statsmodels.api as sm #for statistic models
from random import randrange, uniform #uniform function makes random float values between a range
from scipy import stats #for statistics
from scipy import * #everything from scipy
import scipy.stats
from scipy.stats import ks_2samp # comparison of two samples
from statsmodels.api import qqplot_2samples #for QQ plots
from numpy import * # e.g isnan command
import itertools
import statistics
import os
from os.path import join, isfile
import shutil
import fnmatch 
import sys

#%% 2)IMPORT MODEL DATA
d = 'C://Model data//new_2d//field//Unif//'
[os.path.join(d, o) for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]
directory="C://Model data//3drndfields_norm//NewSimulation_ParameterScan_np_field_r//*distances.csv"
dataframes = []
import glob
all_files2=(glob.glob(directory))
# Create the list for the three DataFrames you want to create:
for filename in all_files2:
    dataframes.append(pd.read_csv(filename))
#% Scaling to experimental frame (Combes et al. 2016)
df_Combes = pd.read_excel('C:/python/' + 'mmc4.xlsx')
# In case column names are misplaces (during calculations)
for i in range(len(dataframes)):
    dataframes[i].rename(columns={'        z_position_(px)':'z_position_(px)'},inplace=True)
	
#%% 3) PEFORM THE CALCULATIONS TO DATA WITH THE BELOW FUCNTIONS:
#SPEEDS
yesx=[]
yesx2=[]
yesx3=[]
yesx=Speed_3D_ci(dataframes,cond='NPCells',conda='pers2',time=101,dt=10,name='C:/python/Model_results/FIELDS/new_fields/speedCI2_NP.csv')
yesx2=Speed_3D_ci(dataframes,cond='ACells',conda='pers2',time=101,dt=10,name='C:/python/Model_results/FIELDS/new_fields/speedCI2_MM.csv')
yesx3=Speed_3D_ci(dataframes,cond='both',conda='pers2',time=101,dt=10,name='C:/python/Model_results/FIELDS/new_fields/speedCI2_NP&MM.csv')
resulti = pd.concat([yesx, yesx2, yesx3], axis=1, sort=False)
resulti.to_csv('C:/python/Model_results/FIELDS/new_fields/speedCI2_all_tikka6819.csv',index=False,header='infer')
#DISTANCES
yes2=[]
yes3=[]
yes4=[]
yes2=Dist_3D_ci(dataframes,time=101,cond='ACells',conda='rnd',name='C:/python/Model_results/FIELDS/new_fields/distCI2_MM.csv')
yes3=Dist_3D_ci(dataframes,time=101,cond='NPCells',conda='rnd',name='C:/python/Model_results/FIELDS/new_fields/distCI2_NP.csv')
yes4=Dist_3D_ci(dataframes,time=101,cond='both', conda='rnd',name='C:/python/Model_results/FIELDS/new_fields/distCI2_NP&MM.csv')
resultia = pd.concat([yes2, yes3, yes4], axis=1, sort=False)
resultia.to_csv('C:/python/Model_results/FIELDS/new_fields/distCI2_all_tikka6819.csv',index=False,header='infer')    
#CONCENTRATIONS (separate run?)
yesxi=[]
yesxi2=[]
yesxi3=[] #check times, also from crit fun
yesxi=conc_3D_ci(dataframes,cond='NPCells',conda='pers2',time=101,name='C:/python/Model_results/FIELDS/new_fields/concCI2_NP.csv')
yesxi2=conc_3D_ci(dataframes,cond='ACells',conda='pers2',time=101,name='C:/python/Model_results/FIELDS/new_fields/concCI2_MM.csv')
yesxi3=conc_3D_ci(dataframes,cond='both',conda='pers2',time=101,name='C:/python/Model_results/FIELDS/new_fields/concCI2_NP&MM.csv')
resultib = pd.concat([yesxi, yesxi2, yesxi3], axis=1, sort=False)
resultib.to_csv('C:/python/Model_results/FIELDS/new_fields/concCI2_all_tikka6819.csv',index=False,header='infer') 
# CELL QUANTITIES
resulti=Amount_3D_ci(dataframes,
                      cond='NPCells',cond1='tip',conda='pers2',time=100,
                      name='C:/python/Model_results/FIELDS/new_fields/UB_NPtip_amount_M7_24919tikka'
                      )    #np_adh3d

resulti= Amount_3D_ci(dataframes,
                      cond='ACells',cond1='tip', conda='pers2',time=100,name='C:/python/Model_results/FIELDS/new_fields/UB_MMtip_amount_M7_24919tikka'
                      )    #np_adh3d

resulti= Amount_3D_ci(dataframes,
                      cond='NPCells',cond1='corner', conda='pers2',time=100,name='C:/python/Model_results/FIELDS/new_fields/UB_NPcorner_amount_M7_24919tikka'
                      )    #np_adh3d
resulti= Amount_3D_ci(dataframes,
                      cond='ACells',cond1='corner', conda='pers2',time=100,name='C:/python/Model_results/FIELDS/new_fields/UB_MMcorner_amount_M7_24919tikka'
                      )    #np_adh3d

#%% 4) CREATE THE FUNCTIONS FOR CALCULATING SPEEDS (LONG DESCRIPTION)  
#First function for speeds
#Select criteria; half of the cells
def crit_fun(dataframes,x0,x1,y0,y1):
    dataframes2 = []
#    crit0=dataframes['time_mcs']==0 #check the times, 26719
    crit1=dataframes['x_position_(px)']>=x0
    crit2=dataframes['y_position_(px)']>=y0
    crit4=dataframes['x_position_(px)']<=x1
    crit5=dataframes['y_position_(px)']<=y1  
    criteria_all=crit1 & crit2 & crit4 & crit5   
    dataframes2=dataframes[criteria_all] #np half (thus h in the name df_dis_nph) 
    return dataframes2        

#Make an auxialiry function for next criteria speed (and for other similarly) function:
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3   

#Select criteria2, e.g. tip or corner:
def speed3_preli(dataframes,cond, conda):
    df_dist_adh3d=dataframes
#% Muokkaus
#% Valitse yhden tiedoston kaikki NP solut yhtena ajan kohtana:
#% Solujen nimet:
    if conda == 'pers':    #check if ok..       
        dfcell_names3d= pd.read_csv('all cells uniform2.csv', delimiter="\t", header=None)
        dfcell_names3da=dfcell_names3d[[0,1]]
    elif conda == 'no-pers':
        dfcell_names3d= pd.read_csv('all cells uniform.csv', delimiter=";", header=None)
        dfcell_names3da=dfcell_names3d[[0,1]]
    elif conda == 'pers2':       
        dfcell_names3d=[]    
        dfcell_names3d=dataframes
        dfcell_names3d=dfcell_names3d.ix[0:195,0:3] #in 2D there are less cells than in 3D
        dfcell_names3d.ix[:,1]=list(np.where(dfcell_names3d.ix[:,1] > 1, 'ACells', 'NPCells'))
        dfcell_names3da=dfcell_names3d.ix[:,0:2] 
        cols = dfcell_names3da.columns.tolist()
        cols = cols[-1:] + cols[:-1]
#%    https://stackoverflow.com/questions/27674880/python-replace-a-number-in-array-with-a-string
    dfoki22=[]
    setti=[]
    settit=[]
    time=[]
    if cond == 'NPCells':
        setti=dfcell_names3da[dfcell_names3da.ix[:,1]==cond].ix[:,0]
    elif cond == 'ACells':
        setti=dfcell_names3da[dfcell_names3da.ix[:,1]==cond].ix[:,0]  
    elif cond == 'both':
        setti=dfcell_names3da.ix[:,0]  
    dfoki22=df_dist_adh3d.set_index("cell_no")
    dfoki22=dfoki22.iloc[setti,:]
    dfoki22=dfoki22.fillna(0)
    settit=list(crit_fun(dfoki22,0,66,0,81).index.unique())    
    #%    if cond == 'corner':
    setti2=intersection(list(setti), settit)   
    if setti2 ==[]:
        setti2=list(setti)
    dataframesa=pd.DataFrame()
    dataframesa=dfoki22
    c1=(40 >= dataframesa['y_position_(px)'])
    dataframesi=dataframesa[c1]
    a=list(dataframesi.index.unique())
    settii=list(setti2) #onko tama setti eri, kuin alla?
    setti22=intersection(settii, a) #ok
    if setti22 ==[]:
        setti22=a 
#%  elif cond == 'tip':
    c3=(40 <= dfoki22['y_position_(px)'])
    dataframesii=pd.DataFrame()
    dataframesii=dfoki22[c3]
    b=list(dataframesii.index.unique())
    setti3=intersection(settii,b) #ok
    if setti3==[]:
        setti3=b
    return settit,setti22, setti3 #both, corner, tip
#%Test might be needed:
#setti, setti2, setti3  = speed3_preli(dataframes[9],cond='NPCells',conda='pers')    #both,corner, tip

#%xyz distances for NPCells, ACells, and Both
def x_y_dis_fun(dataframes,setti,time):
#%The X values:
# Append model values from file with your indeces (with 'for loop') to obtain a list of lists
# https://stackoverflow.com/questions/522563/accessing-the-index-in-python-for-loops
    dfJEE=dataframes.set_index("cell_no") #dfJEE is dataframes
    index = 0
    mmf=[]
    t_jee=time
    for index in setti: 
        mmf.append(np.array(dfJEE.ix[index,'x_position_(px)'])) # mmf[0][-1]: Out[54]: 22.818181818200003 
#% Removing 'nans'
    mmfa=np.zeros((len(setti),t_jee))
    for i in range(len(mmf)):
        for j in range(t_jee):
            if np.shape(mmf[i])==():
                mmfa[i][0]=mmf[i]
            elif  np.shape(mmf[i])==(2,):
                mmfa[i][0]=mmf[i][0]
                mmfa[i][1]=mmf[i][1]
            elif  np.shape(mmf[i])==(3,):
                mmfa[i][0]=mmf[i][0]
                mmfa[i][1]=mmf[i][1]
                mmfa[i][2]=mmf[i][2]
            elif  np.shape(mmf[i])==(4,):
                mmfa[i][0]=mmf[i][0]
                mmfa[i][1]=mmf[i][1]
                mmfa[i][2]=mmf[i][2]
                mmfa[i][3]=mmf[i][3]           
            elif  np.shape(mmf[i])==(5,):
                mmfa[i][0]=mmf[i][0]
                mmfa[i][1]=mmf[i][1]
                mmfa[i][2]=mmf[i][2]
                mmfa[i][3]=mmf[i][3]
                mmfa[i][4]=mmf[i][4]
            elif  int(np.asarray(mmf[i].shape))<101:
                for k in range(int(np.asarray(mmf[i].shape))):
                    mmfa[i][k]=mmf[i][k]    
            elif  np.shape(mmf[i])==(j,):
                for k in range(j):
                    mmfa[i][k]=mmf[i][k]             
            else:
               mmfa[i]=mmf[i]   
    mmf=[]
    mmf=mmfa             
    for i in range(len(mmf)):
        for j in range(len(mmf[i])):
            if mmf[i][j] == 0:
                lenin=int(round(len(mmf[i])/4)) # latest previous values
                mmf[i][j] = mmf[i][j-1] # replacing with mean of latest previous values     
# The symmetrical numpy matrix made from this list of list by padding constant values to the end 
# of those lists that are below the amount of cells 
    mtt=np.zeros((len(setti),t_jee))
    if mtt.shape == mmf.shape:
        mtt=mmf
    elif mtt.shape != mmf.shape:    
        for i in range(len(mmf)):
            Ashape=mmf[i].shape
            new_shape=t_jee 
            shape_diff = np.asarray(new_shape) - np.asarray(Ashape)
            if shape_diff==0:
                mtt[i,:]=np.lib.pad(mmf[i], (0,shape_diff), 'constant', constant_values=(0))
    index = 0
    mmf=[]
    t_jee=time
    for index in setti: 
        mmf.append(np.array(dfJEE.ix[index,'y_position_(px)'])) # mmf[0][-1]: Out[54]: 22.818181818200003 
#% Removing 'nans'
    mmfa=np.zeros((len(setti),t_jee))
    for i in range(len(mmf)):
        for j in range(t_jee):
            if np.shape(mmf[i])==():
                mmfa[i][0]=mmf[i]
            elif  np.shape(mmf[i])==(2,):
                mmfa[i][0]=mmf[i][0]
                mmfa[i][1]=mmf[i][1]
            elif  np.shape(mmf[i])==(3,):
                mmfa[i][0]=mmf[i][0]
                mmfa[i][1]=mmf[i][1]
                mmfa[i][2]=mmf[i][2]
            elif  np.shape(mmf[i])==(4,):
                mmfa[i][0]=mmf[i][0]
                mmfa[i][1]=mmf[i][1]
                mmfa[i][2]=mmf[i][2]
                mmfa[i][3]=mmf[i][3]
            elif  np.shape(mmf[i])==(5,):
                mmfa[i][0]=mmf[i][0]
                mmfa[i][1]=mmf[i][1]
                mmfa[i][2]=mmf[i][2]
                mmfa[i][3]=mmf[i][3]
                mmfa[i][4]=mmf[i][4]
            elif  int(np.asarray(mmf[i].shape))<101:
                for k in range(int(np.asarray(mmf[i].shape))):
                    mmfa[i][k]=mmf[i][k]    
            elif  np.shape(mmf[i])==(j,):
                for k in range(j):
                    mmfa[i][k]=mmf[i][k]         
            else:
               mmfa[i]=mmf[i]
    mmf=[]
    mmf=mmfa   
    for i in range(len(mmf)):
        mmf[i][np.where(np.isnan(mmf[i]))]=0
        for j in range(len(mmf[i])):
            if mmf[i][j] == 0:
                lenin=int(round(len(mmf[i])/4)) # latest previous values
                mmf[i][j] = mmf[i][j-1] # replacing with mean of latest previous values       
# The symmetrical numpy matrix made from this list of list by padding constant values to the end 
# of those lists that are below the amount of cells 
    mtt2=np.zeros((len(setti),t_jee))
    if mtt2.shape == mmf.shape:
        mtt2=mmf
    elif mtt.shape != mmf.shape: 
        for i in range(len(mmf)):
            Ashape=mmf[i].shape
            new_shape=t_jee 
            shape_diff = np.asarray(new_shape) - np.asarray(Ashape)
            mtt2[i,:]=np.lib.pad(mmf[i], (0,shape_diff), 'constant', constant_values=(0))
    return mtt, mtt2
#% Test is always a good thing, also here in the functions sections:
#    axan, byan= x_y_dis_fun(dataframes[9],setti,time=100)

# New speed function
def news1(dataframes,setti,time,dt):
    axan, byan = x_y_dis_fun(dataframes,setti,time)
    #Selecting the correct spaces
    index_cells=len(setti)            # max cell values
    index_time=time
    r=np.zeros((index_cells,index_time)) # here should be the size of the matrix len(...)
    for j in range(index_time-1):
        r[:,j] = np.sqrt((axan[:,(j+1)]-axan[:,j])**2+(byan[:,(j+1)]-byan[:,j])**2)
    r2=r/dt
    r2=r2[:,0:(time-1)]
    
    return r2
#Test:
#r2=  news1(dataframes[9],setti,time=100,dt=10) 

#% 3D speed arvot matriisina (kaikki samassa, kulmalle, kärjelle ja kärki per kulma-arvoille), load X_y_z.. and inst_fun
def speed_fun(dataframes,setti,setti2,setti3,time,dt):   #both,corner,tip
#% the values from df_dis_ub3dh, i.e. ub half (h), but can be all 
    from random import randrange, uniform #just in case here, needed for random indexing  
#    cond='corner'
    speed=[]
    yes=[]
    speed2=[]
    yes2=[]
    speed3=[]
    yes3=[]
# https://stackoverflow.com/questions/522563/accessing-the-index-in-python-for-loops    
#%    elif cond == 'both':              
    speed3=news1(dataframes,setti,time,dt)  
#%  elif cond == 'corner':
    speed2=news1(dataframes,setti2,time,dt)      
#	tip
    speed=news1(dataframes,setti3,time,dt)      
    yes=pd.DataFrame(speed)    #tip
    yes2=pd.DataFrame(speed2) #corner
    yes3=pd.DataFrame(speed3) #both                
    return yes, yes2, yes3 #tip,corner,both speeds
#Test:
#    yes, yes2, yes3=speed_fun(dataframes[0],setti,setti2,setti3,time=101,dt=10)

#Important confidennce interval function for the next and other calculation function, here though separately:
#https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data        
def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

	return m, m-h, m+h

#% 3D speed CI:s    
def Speed_3D_ci(dataframes,cond,conda,time,dt,name):
    tot=[]
    tot2=[]
    for i in range(len(dataframes)):
        tot.append(speed3_preli(dataframes[i],cond,conda))
        tot2.append(speed_fun(
                dataframes[i],
                tot[i][0],
                tot[i][1],
                tot[i][2],
                time,dt))
    frames=[]
    frames2=[]
    frames3=[]
    for i in range(len(dataframes)):
        frames.append(tot2[i][0])
        frames2.append(tot2[i][1])
        frames3.append(tot2[i][2])
    result = pd.concat(frames)
    result2 = pd.concat(frames2)
    result3 = pd.concat(frames3)
#https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
#% Laskeminen (CI)
    yes=[]
    yes2=[]
    yes3=[]
    for i in range(time-1):
        yes.append(mean_confidence_interval(result[[i]], confidence=0.95))
        yes2.append(mean_confidence_interval(result2[[i]], confidence=0.95))
        yes3.append(mean_confidence_interval(result3[[i]], confidence=0.95))        
#    #eli taa on yhdelle riville... tais olla siina.. :)
    yes=pd.DataFrame(yes)
    yes2=pd.DataFrame(yes2) 
    yes3=pd.DataFrame(yes3) 
   
    for i in range(time-1):
        yes.loc[i,1]=float(yes.loc[i,1])
        yes.loc[i,2]=float(yes.loc[i,2])
        yes2.loc[i,1]=float(yes2.loc[i,1])
        yes2.loc[i,2]=float(yes2.loc[i,2])
        yes3.loc[i,1]=float(yes3.loc[i,1])
        yes3.loc[i,2]=float(yes3.loc[i,2]) 
##or https://stackoverflow.com/questions/44603615/plot-95-confidence-interval-errorbar-python-pandas-dataframes
##% Kääntö
    yes[1] = yes[1].astype(float)
    yes[2] = yes[2].astype(float) 
    yes2[1] = yes2[1].astype(float)
    yes2[2] = yes2[2].astype(float) 
    yes3[1] = yes3[1].astype(float)
    yes3[2] = yes3[2].astype(float) 
    yes3[3]=tuple(range(time-1))
    resulti = pd.concat([yes, yes2, yes3], axis=1, sort=False)
#https://datatofish.com/convert-string-to-float-dataframe/ 
    resulti.columns = [
                 'Speed_tip_avg', 'Speed_tip_CI_min', 'Speed_tip_CI_max',
                 'Speed_corner_avg', 'Speed_corner_CI_min', 'Speed_corner_CI_max',
                 'Speed_overall_avg', 'Speed_overall_CI_min', 'Speed_overall_CI_max',
                 'Time (MCS)']   
    
    resulti = resulti[['Time (MCS)',                 
                 'Speed_tip_avg', 'Speed_tip_CI_min', 'Speed_tip_CI_max',
                 'Speed_corner_avg', 'Speed_corner_CI_min', 'Speed_corner_CI_max',
                 'Speed_overall_avg', 'Speed_overall_CI_min', 'Speed_overall_CI_max']]
#%https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas
    resulti.to_csv(name,index=False,header='infer') #Tallennus samassa
    return resulti

#%% 5) CREATE THE FUNCTIONS FOR CALCULATING DISTANCES (LONG DESTCTIPTIONSTCTIPTION)  
#First function for distances
#Select criteria; half of the cells
def crit_fun(dataframes,x0,x1,y0,y1):
    dataframes2 = []
    crit0=dataframes['time_mcs']==0 #check the times, 26719
    crit1=dataframes['x_position_(px)']>=x0
    crit2=dataframes['y_position_(px)']>=y0
    crit4=dataframes['x_position_(px)']<=x1
    crit5=dataframes['y_position_(px)']<=y1  
    criteria_all=crit0 & crit1 & crit2 & crit4 & crit5   
    dataframes2=dataframes[criteria_all] #np half (thus h in the name df_dis_nph) 
    return dataframes2    

#%   3D distance arvot listana, (myöhemmin matriisina, josta CI lasku)
def dist3_preli(dataframes,cond,conda):
    df_dist_adh3d=dataframes
#% Valitse yhden tiedoston kaikki NP solut yhtena ajan kohtana:
# Name of the cells:
    if conda == 'rnd':       
        dfcell_names3d=[]    
        dfcell_names3d=df_dist_adh3d   
        dfcell_names3d=dfcell_names3d.ix[0:195,0:2]
        dfcell_names3d.ix[:,1]=list(np.where(dfcell_names3d.ix[:,1] > 1, 'ACells', 'NPCells'))
        dfcell_names3da=dfcell_names3d.ix[:,0:2] 
    elif conda == 'norm':
        dfcell_names3d= pd.read_csv('all cells uniform.csv', delimiter=";", header=None)
        dfcell_names3da=dfcell_names3d[[0,1]]            
#% selecting common indeces in the list of lists of indeces 
    dfoki22=[]
    setti=[]
    settit=[]
    time=[]
    if cond == 'both':           
        dfcell_names3da.columns=['a','b']
        c1=(dfcell_names3da['b'] == 'NPCells')
        c2=(dfcell_names3da['b'] == 'ACells')
        cond=c1+c2
        setti=dfcell_names3da.loc[cond,"a"] #all
    else:
        setti=dfcell_names3da[dfcell_names3da.ix[:,1]==cond].ix[:,0] #riittää määrää, len(setti)=196
    dfoki22=df_dist_adh3d.set_index("cell_no")
    dfoki22=dfoki22.loc[list(setti),:]
    dfoki22=dfoki22.dropna()
    time=len((dfoki22.set_index("time_mcs")).index.unique()) #length
    dfoki22=dfoki22.loc[list(crit_fun(dfoki22,0,56,0,81).index.unique()),:] #39,61, ok ->104 cells
    settia=list(crit_fun(dfoki22,0,56,0,81).index.unique())
    if settia ==[]:
        settia=setti 
    return dfcell_names3da,settia,time 
#Test is good to have:
#dfcell_names3da,settia,time=dist3_preli(dataframes[7],cond='NPCells',conda='rnd')

def dishi_fun(dataframes,dfcell_names3da,settia,time,df_Combes):   
# Append model values from file with your indeces (with 'for loop') to obtain a list of lists
# You do not necessary need a range (0,1,2,...), but indeces 8,99,77 etc. for your loop.
# https://stackoverflow.com/questions/522563/accessing-the-index-in-python-for-loops
# Also, panda idexing with .ix, not obvious 
#% https://stackoverflow.com/questions/522563/accessing-the-index-in-python-for-loops
#    if cond == 'corner':
    c1=(dataframes['y_position_(px)'] <= 40)
    dataframesi=dataframes[c1]
    dataframesi=dataframesi.set_index("cell_no")
    a=list(dataframesi.index.unique())
    setti2=intersection(a, settia) #ok
    if setti2 ==[]:
        setti2=a   
#    elif cond == 'tip':
    c2=(dataframes['y_position_(px)'] >= 40)
    dataframesii=dataframes[c2]
    dataframesii=dataframesii.set_index("cell_no")
    b=list(dataframesii.index.unique())
    setti3=intersection(b, settia) #ok    
    if setti3 ==[]:
        setti3=b        
    index=0
    mmf=[]
    mmf2=[]
    for index in setti2:
        mmf.append(np.array(dataframesi.loc[index,'cell_dist_to_corner'])) 
        #dataframes[0].ix[33,2] #corner
    index=0
    for index in setti3:    
        mmf2.append(np.array(dataframesii.loc[index,'cell_dist_to_tip'])) #tip
#        https://codereview.stackexchange.com/questions/156447/subtract-multiple-columns-in-pandas-dataframe-by-a-series-single-column
# The symmetrical numpy matrix made from this list of list by padding constant values to the end 
# of those lists that are below the amount of cells 
    mtt=np.zeros((len(mmf),time)) #test also: len(dataframes[0].ix[:,1].unique())
    mt2=np.zeros((len(mmf2),time)) #test also: len(dataframes[0].ix[:,1].unique())%
    lenin2=[]
    lenin22=[]
    lenin23=[]
    for i in range(len(mmf)):
        Ashape=mmf[i].size #size is better than shape
        new_shape=time
        shape_diff = np.asarray(new_shape) - np.asarray(Ashape)
        mtt[i,:]=np.lib.pad(mmf[i], (0,shape_diff), 'constant', constant_values=(0.1))
    for i in range(len(mmf2)):    
        Ashape2=mmf2[i].size #size is better than shape
        new_shape2=time
        shape_diff2 = np.asarray(new_shape2) - np.asarray(Ashape2)
        mt2[i,:]=np.lib.pad(mmf2[i], (0,shape_diff2), 'constant', constant_values=(0.1))
    for i in range(len(mmf)):
        if np.where(np.isnan(mmf[i])) == True:
            mmf[i][np.where(np.isnan(mmf[i]))]=+0.1
        for j in range(mmf[i].size):
            if np.atleast_1d(mmf[i])[j] == +0.1: #note the np.atleast_id
                lenin2=int(round(len(mmf[i])/4)) # latest previous values
                mmf[i][j] == abs(mean(mmf[i][-lenin2:-2])) # replacing with mean of latest previous values                
    for i in range(len(mmf2)):
        if np.where(np.isnan(mmf2[i])) == True:            
            mmf2[i][np.where(np.isnan(mmf2[i]))]=+0.1
        for j in range(mmf2[i].size):
            if np.atleast_1d(mmf2[i])[j] == +0.1: #note the np.atleast_id
        #https://stackoverflow.com/questions/25458553/is-there-a-pythonic-way-to-change-scalar-and-0d-array-to-1d-array?rq=1&utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
                lenin22=int(round(len(mmf2[i])/4)) # latest previous values
                mmf2[i][j] == abs(mean(mmf2[i][-lenin22:-2])) # replacing w
# And finally changing those constant values (0.1) to the random values close to the end values
#% These are for the ranges
    uli=mtt.shape[0]
    uli2=mtt.shape[1]
    for i in range(uli):
        for j in range(uli2):
            if mtt[i,j]==0.1:
                mtt[i,j]=mmf[i][-1]+uniform(-1,1)*std(mtt[i,:])*0.1
    ulii=mt2.shape[0]
    ulii2=mt2.shape[1]
    for i in range(ulii):
        for j in range(ulii2):
            if mt2[i,j]==0.1:
                mt2[i,j]=mmf2[i][-1]+uniform(-1,1)*std(mt2[i,:])*0.1
# https://stackoverflow.com/questions/3996904/generate-random-integers-between-0-and-9  
    array = dfcell_names3da.ix[:,0]
    item = list(setti2)
    lista=[]
    for i in range(len(item)):
        lista.append((np.where(array.ix[0:]==item[i]))[0][0])
    row_names=dfcell_names3da.ix[lista,1]
    row_names=list(row_names) #row was one...   
    #%
    array2 = dfcell_names3da.ix[:,0]
    item2 = list(setti3)
    lista2=[]
    for i in range(len(item2)):
        lista2.append((np.where(array2.ix[0:]==item2[i]))[0][0])
    row_names2=dfcell_names3da.ix[lista2,1]
    row_names2=list(row_names2) #row was one... 
#https://stackoverflow.com/questions/176918/finding-the-index-of-an-item-given-a-list-containing-it-in-python
#related: https://stackoverflow.com/questions/2864842/common-elements-comparison-between-2-lists
#https://www.geeksforgeeks.org/python-intersection-two-lists/
    mtt = pd.DataFrame(mtt, columns=None, index=row_names)
    mt2 = pd.DataFrame(mt2, columns=None, index=row_names2)
    dataframesa=[]
    dataframesb=[]
    dataframes=[]
    dataframesi=[]
    dataframesii=[]
    return mtt, mt2 #corner and tip (region) distances, tip/corner (above)    
#Test is good:   
#mtt, mt2=dishi_fun(dataframes[7],dfcell_names3da,settia,time,df_Combes)

#% 3D distance CI:s    
def Dist_3D_ci(dataframes,time,cond,conda,name):
    tot  = []
    tot2 = []
    for i in range(len(dataframes)):
        tot.append(dist3_preli(dataframes[i],cond,conda))
        tot2.append(dishi_fun(dataframes[i],tot[i][0],tot[i][1],time,df_Combes)) #ub half.. 
    frames=[]
    frames2=[]
    frames3=[]
    for i in range(len(dataframes)):
        frames.append(tot2[i][0])
        frames2.append(tot2[i][1])
    result = pd.concat(frames)
    result2 = pd.concat(frames2)
#https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html       
    yes=[]
    yes2=[]
    for i in range(time):
        yes.append(mean_confidence_interval(result[[i]], confidence=0.95))
        yes2.append(mean_confidence_interval(result2[[i]], confidence=0.95))        
    yes=pd.DataFrame(yes)
    yes2=pd.DataFrame(yes2)     
    for i in range(time):
        yes.loc[i,1]=float(yes.loc[i,1])
        yes.loc[i,2]=float(yes.loc[i,2])
        yes2.loc[i,1]=float(yes2.loc[i,1])
        yes2.loc[i,2]=float(yes2.loc[i,2])
#or https://stackoverflow.com/questions/44603615/plot-95-confidence-interval-errorbar-python-pandas-dataframes
#% Kääntö
    yes[1] = yes[1].astype(float)
    yes[2] = yes[2].astype(float) 
    yes2[1] = yes2[1].astype(float)
    yes2[2] = yes2[2].astype(float)   
    yes2[3]=tuple(range(time))
    resulti = pd.concat([yes, yes2], axis=1, sort=False)
#https://datatofish.com/convert-string-to-float-dataframe/ 
    resulti.columns = ['Dist_corner_avg', 'Dist_corner_CI_min', 'Dist_corner_CI_max',
                 'Dist_tip_avg', 'Dist_tip_CI_min', 'Dist_tip_CI_max', 'Time (MCS)']  
    resulti = resulti[['Time (MCS)','Dist_corner_avg', 'Dist_corner_CI_min', 'Dist_corner_CI_max',
                 'Dist_tip_avg', 'Dist_tip_CI_min', 'Dist_tip_CI_max']]
#https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas
    resulti.to_csv(name, index=False, header='infer') #Tallennus samassa
    return resulti

#%% 7) CREATE THE FUNCTIONS FOR CALCULATING CONCENTRATIONS (LONG DESCRIPTION)  
#First function concentrations, the half criteria:
def crit_fun(dataframes,x0,x1,y0,y1):
    dataframes2 = []
    crit0=dataframes['time_mcs']==0 #check the times, 26719
    crit1=dataframes['x_position_(px)']>=x0
    crit2=dataframes['y_position_(px)']>=y0
    crit4=dataframes['x_position_(px)']<=x1
    crit5=dataframes['y_position_(px)']<=y1  
    criteria_all=crit0 & crit1 & crit2 & crit4 & crit5   
    dataframes2=dataframes[criteria_all] #np half (thus h in the name df_dis_nph) 
    return dataframes2     
    
#%   Valitaan oikeat solut listana, (myöhemmin matriisina, josta CI lasku)
def conc3_preli(dataframes,cond,conda):
    df_dist_adh3d=dataframes
#% Muokkaus
#% Valitse yhden tiedoston kaikki NP solut yhtena ajan kohtana:
#% Solujen nimet:
#    conda='pers2'
    if conda == 'pers':    #check if ok..       
        dfcell_names3d= pd.read_csv('all cells uniform3d_v3a.csv', delimiter="\t", header=None)
        dfcell_names3da=dfcell_names3d[[0,1]]
    elif conda == 'no-pers':
        dfcell_names3d= pd.read_csv('all cells uniform.csv', delimiter=";", header=None)
        dfcell_names3da=dfcell_names3d[[0,1]]
    elif conda == 'pers2':       
        dfcell_names3d=[]    
        dfcell_names3d=dataframes
        dfcell_names3d=dfcell_names3d.ix[0:195,0:3]
        dfcell_names3d.ix[:,1]=list(np.where(dfcell_names3d.ix[:,1] > 1, 'ACells', 'NPCells'))
        dfcell_names3da=dfcell_names3d.ix[:,0:2] 
        cols = dfcell_names3da.columns.tolist()
        cols = cols[-1:] + cols[:-1]
#%    https://stackoverflow.com/questions/27674880/python-replace-a-number-in-array-with-a-string
    dfoki22=[]
    setti=[]
    settit=[]
    time=[]
#    cond = 'NPCells'
    if cond == 'NPCells':
        setti=dfcell_names3da[dfcell_names3da.ix[:,1]==cond].ix[:,0]
    elif cond == 'ACells':
        setti=dfcell_names3da[dfcell_names3da.ix[:,1]==cond].ix[:,0] 
    elif cond == 'both':
        setti=dfcell_names3da.ix[:,0]  
    dfoki22=df_dist_adh3d.set_index("cell_no")
    dfoki22=dfoki22.loc[list(setti),:]
    dfoki22=dfoki22.dropna() 
    #%#%    elif cond == 'both':  
    #39,61, ok ->104 cells?? more likely.. 196 no, it is 104, since around half amount of cells
    settit=list(crit_fun(dfoki22,0,66,0,81).index.unique())
    #%    if cond == 'corner':
    setti2=intersection(setti, settit)
    if setti2==[]:
        setti2=setti
    dataframesa=pd.DataFrame()
    dataframesa=dfoki22
    c1=(40 >= dataframesa['y_position_(px)'])
    dataframesi=dataframesa[c1]
    a=list(dataframesi.index.unique())
    settii=list(setti2) #onko tama setti eri, kuin alla?
    setti22=intersection(settii, a) #ok
    if setti22 ==[]:
        setti22=a
#%  elif cond == 'tip':
    c3=(40 <= dfoki22['y_position_(px)'])
    dataframesii=pd.DataFrame()
    dataframesii=dfoki22[c3]
    b=list(dataframesii.index.unique())
#    settiii=list(setti2)
    setti3=intersection(settii,b) #ok
    if setti3 ==[]:
        setti3=b
    return settit,setti22, setti3 #both, corner, tip
#% Test is good
#setti, setti2, setti3  = conc3_preli(dataframes[3],cond='both',conda='pers2')    #both,corner, tip

#%xyz dis for NPCells, ACells, and Both
def x_y_z_dis_fun2(dataframes,setti,time):
    from random import randrange, uniform #just in case here, needed for random indexing    
#%The X values:
# https://stackoverflow.com/questions/522563/accessing-the-index-in-python-for-loops
    dfJEE=dataframes.set_index("cell_no")
    index = 0
    mmf=[]
    for index in setti: 
        mmf.append(np.array(dfJEE.ix[index,'cell_field'])) # mmf[0][-1]: Out[54]: 22.818181818200003
    mmfa=np.zeros((len(setti),time))
    for i in range(len(mmf)):
        for j in range(time):
            if np.shape(mmf[i])==():
                mmfa[i][0]=mmf[i]
            elif  np.shape(mmf[i])==(2,):
                mmfa[i][0]=mmf[i][0]
                mmfa[i][1]=mmf[i][1]
            elif  np.shape(mmf[i])==(3,):
                mmfa[i][0]=mmf[i][0]
                mmfa[i][1]=mmf[i][1]
                mmfa[i][2]=mmf[i][2]
            elif  np.shape(mmf[i])==(4,):
                mmfa[i][0]=mmf[i][0]
                mmfa[i][1]=mmf[i][1]
                mmfa[i][2]=mmf[i][2]
                mmfa[i][3]=mmf[i][3]         
            elif  np.shape(mmf[i])==(5,):
                mmfa[i][0]=mmf[i][0]
                mmfa[i][1]=mmf[i][1]
                mmfa[i][2]=mmf[i][2]
                mmfa[i][3]=mmf[i][3]
                mmfa[i][4]=mmf[i][4]
            elif  int(np.asarray(mmf[i].shape))<101:
                for k in range(int(np.asarray(mmf[i].shape))):
                    mmfa[i][k]=mmf[i][k]    
            elif  np.shape(mmf[i])==(j,):
                for k in range(j):
                    mmfa[i][k]=mmf[i][k]         
            else:
               mmfa[i]=mmf[i]
    mmf=[]
    mmf=mmfa     
    for i in range(len(mmf)):
        for j in range(len(mmf[i])):
            if mmf[i][j] == 0:
                lenin=int(round(len(mmf[i])/4)) # latest previous values
                mmf[i][j] = mmf[i][j-1] # replacing with mean of latest previous values
    t_jee=time   
# The symmetrical numpy matrix made from this list of list by padding constant values to the end 
# of those lists that are below the amount of cells 
    mtt=np.zeros((len(setti),time))   
    if mtt.shape == mmf.shape:
        mtt=mmf
    elif mtt.shape != mmf.shape: 
        for i in range(len(mmf)):
            Ashape=mmf[i].shape
            new_shape=t_jee 
            shape_diff = np.asarray(new_shape) - np.asarray(Ashape)
            mtt[i,:]=np.lib.pad(mmf[i], (0,shape_diff), 'constant', constant_values=(0.1))   
# And finally changing those constant values (0.1) to the random values close to the end values
# These are for the ranges
    uli=mtt.shape[0]
    uli2=mtt.shape[1]
    for i in range(uli):
        for j in range(uli2):
            if mtt[i,j]==0.1:
                lenin2=int(round(len(mmf[i])/4)) # final std range (the fourth quartile of last values)
                mtt[i,j]=mmf[i][-1]+uniform(-1,1)*std(mmf[i][-lenin2:-1])*0.25
    return mtt

#% New concentration function
def news(dataframes,setti,time):
    axan= x_y_z_dis_fun2(dataframes,setti,time)
    #Selecting the correct spaces
    index_cells=len(setti)            # max cell values
    index_time=time
    r=np.zeros((index_cells,index_time)) # here should be the size of the matrix len(...)
    for j in range(index_time-1):
        r[:,j] = abs((axan[:,(j+1)]-axan[:,j]))
    r2=r[:,0:(time-1)]   
    return r2
#%Test is good:
#    r2=news(dataframes[0],setti,time=101)
    
#% 3D conc arvot matriisina (kaikki samassa, kulmalle, kärjelle ja kärki per kulma-arvoille), load X_y_z.. and inst_fun
def conc_fun(dataframes,setti,setti2,setti3,time):   #both,corner,tip
#% the values from df_dis_ub3dh, i.e. ub half (h), but can be all 
    from random import randrange, uniform #just in case here, needed for random indexing  
#    cond='corner'
    conc=[]
    yes=[]
    conc2=[]
    yes2=[]
    conc3=[]
    yes3=[]
# https://stackoverflow.com/questions/522563/accessing-the-index-in-python-for-loops    
#%    elif cond == 'both':              
    conc3=news(dataframes,setti,time) 
#%  elif cond == 'corner':
    conc2=news(dataframes,setti2,time)    
#tip
    conc=news(dataframes,setti3,time)     
    yes=pd.DataFrame(conc)    #tip
    yes2=pd.DataFrame(conc2) #corner
    yes3=pd.DataFrame(conc3) #both                   
    return yes, yes2, yes3 #tip,corner,both concs
 
#% 3D speed CI:s    
def conc_3D_ci(dataframes,cond,conda,time,name):
    tot=[]
    tot2=[]
    for i in range(len(dataframes)):
        tot.append(conc3_preli(dataframes[i],cond,conda))
        tot2.append(conc_fun(
                dataframes[i],
                tot[i][0],
                tot[i][1],
                tot[i][2],
                time))
    frames=[]
    frames2=[]
    frames3=[]
    for i in range(len(dataframes)):
        frames.append(tot2[i][0])
        frames2.append(tot2[i][1])
        frames3.append(tot2[i][2])
    result = pd.concat(frames)
    result2 = pd.concat(frames2)
    result3 = pd.concat(frames3)
#https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html  
    yes=[]
    yes2=[]
    yes3=[]
    for i in range(time-1):
        yes.append(mean_confidence_interval(result[[i]], confidence=0.95))
        yes2.append(mean_confidence_interval(result2[[i]], confidence=0.95))
        yes3.append(mean_confidence_interval(result3[[i]], confidence=0.95))
    yes=pd.DataFrame(yes)
    yes2=pd.DataFrame(yes2) 
    yes3=pd.DataFrame(yes3)   
    for i in range(time-1):
        yes.loc[i,1]=float(yes.loc[i,1])
        yes.loc[i,2]=float(yes.loc[i,2])
        yes2.loc[i,1]=float(yes2.loc[i,1])
        yes2.loc[i,2]=float(yes2.loc[i,2])
        yes3.loc[i,1]=float(yes3.loc[i,1])
        yes3.loc[i,2]=float(yes3.loc[i,2])
##or https://stackoverflow.com/questions/44603615/plot-95-confidence-interval-errorbar-python-pandas-dataframes
##% Kääntö
    yes[1] = yes[1].astype(float)
    yes[2] = yes[2].astype(float) 
    yes2[1] = yes2[1].astype(float)
    yes2[2] = yes2[2].astype(float) 
    yes3[1] = yes3[1].astype(float)
    yes3[2] = yes3[2].astype(float) 
    yes3[3]=tuple(range(time-1))
    resulti = pd.concat([yes, yes2, yes3], axis=1, sort=False)
#https://datatofish.com/convert-string-to-float-dataframe/ #%
    resulti.columns = [
                 'conc_tip_avg', 'conc_tip_CI_min', 'conc_tip_CI_max',
                 'conc_corner_avg', 'conc_corner_CI_min', 'conc_corner_CI_max',
                 'conc_overall_avg', 'conc_overall_CI_min', 'conc_overall_CI_max',
                 'Time (MCS)']
    resulti = resulti[['Time (MCS)',                 
                 'conc_tip_avg', 'conc_tip_CI_min', 'conc_tip_CI_max',
                 'conc_corner_avg', 'conc_corner_CI_min', 'conc_corner_CI_max',
                 'conc_overall_avg', 'conc_overall_CI_min', 'conc_overall_CI_max']]
#%https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas
    resulti.to_csv(name,index=False,header='infer') #Tallennus samassa
    return resulti
	
#%% 7) CREATE THE FUNCTIONS FOR CALCULATING QUANTITIES (LONG DESCRIPTION)  
#The first funciton for the conditions, corner or tip region:
def crit_funt(dataframes,y0, cond1):
    dataframes2 = []
    if cond1 == 'tip': 
        crit2=dataframes['y_position_(px)']>=y0 #'corner' or 'tip, above 40'
    elif cond1 == 'corner': 
        crit2=dataframes['y_position_(px)']<=y0 #'corner', below 40 or 'tip, above 40'
    dataframes2=dataframes[crit2]
    return dataframes2 

#The quantitiy or cell amount function:
def amount(dataframes, cond, cond1, conda):
#    cond1='corner'
    dataframes2=crit_funt(dataframes,40,cond1)
    df_dist_adh3d=dataframes2
#    conda='pers2'
    if conda == 'pers':    #check if ok..       
        dfcell_names3d= pd.read_csv('all cells uniform3d_v3a.csv', delimiter="\t", header=None)
        dfcell_names3da=dfcell_names3d[[0,1]]
    elif conda == 'no-pers':
        dfcell_names3d= pd.read_csv('all cells uniform.csv', delimiter=";", header=None)
        dfcell_names3da=dfcell_names3d[[0,1]]
    elif conda == 'pers2':       
        dfcell_names3d=[]    
        dfcell_names3d=dataframes
        dfcell_names3d=dfcell_names3d.iloc[0:195,0:3]
        dfcell_names3d.ix[:,1]=list(np.where(dfcell_names3d.ix[:,1] > 1, 'ACells', 'NPCells'))
        dfcell_names3da=dfcell_names3d.ix[:,0:2] 
        cols = dfcell_names3da.columns.tolist()
        cols = cols[-1:] + cols[:-1]
#%    https://stackoverflow.com/questions/27674880/python-replace-a-number-in-array-with-a-string
    dfoki22=[]
    setti=[]
    settit=[]
    time=[]
#    cond = 'NPCells'
    if cond == 'NPCells':
        setti=dfcell_names3da[dfcell_names3da.ix[:,1]==cond].ix[:,0]
    elif cond == 'ACells':
        setti=dfcell_names3da[dfcell_names3da.ix[:,1]==cond].ix[:,0]  
        #%riittää määrää, len(setti)=196  
    dfoki22=df_dist_adh3d.set_index("cell_no")
    dfoki22=dfoki22.loc[list(setti),:]
    dfoki22=dfoki22.dropna()
#    https://stackoverflow.com/questions/2161752/how-to-count-the-frequency-of-the-elements-in-a-list/2162045
#    https://stackoverflow.com/questions/41217310/get-index-of-a-row-of-a-pandas-dataframe-as-an-integer/42853445
    a=dfoki22['time_mcs']
    from itertools import groupby
    results = {value: len(list(freq)) for value, freq in groupby(sorted(a))}
    #%
    return results
	
#% 3D amount CI:s, jos aikojen maara eri kuin 101, niin muuta    
def Amount_3D_ci(dataframes,cond,cond1,conda,time,name):
    tot2 = []
    for i in range(len(dataframes)):
        tot2.append(amount(dataframes[i],cond,cond1,conda)) #ub half.. 
    frames2=[]
    for i in range(len(dataframes)):
        frames2.append(pd.DataFrame(([v for v in tot2[i].values()])))
#https://www.quora.com/How-do-I-convert-a-dictionary-to-a-list-in-Python 
    result2 = pd.concat(frames2, axis=1)
#https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html      
    yes=[]
    for i in range(time):
        yes.append(mean_confidence_interval(result2.ix[i], confidence=0.95))
    yes=pd.DataFrame(yes)
    for i in range(time):
        yes.loc[i,1]=float(yes.loc[i,1])
        yes.loc[i,2]=float(yes.loc[i,2])
#or https://stackoverflow.com/questions/44603615/plot-95-confidence-interval-errorbar-python-pandas-dataframes
#% Kääntö
    yes[1] = yes[1].astype(float)
    yes[2] = yes[2].astype(float) 
    resulti = pd.concat([yes], axis=1, sort=False)
#https://datatofish.com/convert-string-to-float-dataframe/ 
    resulti.columns = ['Cells_region_avg', 'Cells_region_CI_min', 'Cells_region_CI_max']  
#https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas
    resulti.to_csv(name, index=False, header='infer') #Tallennus samassa    
    return resulti   
#%%Drive the amount function:
Amount_3D_ci(dataframes,cond = 'ACells',cond1='corner' ,conda='pers2',time=101,
             name='C:/python/Model_results/2D_new/Rand/UB_MMtip_amount_24919tikka')
#%%Rename (and move model simulation) files (yes, three phases to get the from A to B)
#https://stackoverflow.com/questions/38948774/python-append-folder-name-to-filenames-in-all-sub-folders

#%% 8) BRING OR CURATE FILES FROM OTHER FOLDERS TO ANTOHER INORDER TO GET THEM IN OK SHAPE FOR CALCULATIONS:
#Auxialiry funciton for next 'remove' function:
def copyfiles(srcdir, dstdir, filepattern):
	def failed(exc):
		raise exc
	for dirpath, dirs, files in os.walk(srcdir, topdown=True, onerror=failed):
		for file in fnmatch.filter(files, filepattern):
			shutil.move(os.path.join(dirpath, file), dstdir)
		break # no recursion
		
#Remove somethinging from folders:
def renmov(root_dir = 'C://Users//pauli//CC3DWorkspace//Random results//NewSimulation_ParameterScan_adh_adh',
           dest_dir = 'C:/Model data/3dr/',
           dest_final = 'C://Model data//3dr//adh_adh3d//'):
    #change the operation's directory "C://Users//..//NewSimulation_XYZ" XYZ as you require
    for root, dirs, files in os.walk(root_dir):
        if not files:
            continue
        prefix = os.path.basename(root)
        for f in files:
            os.rename(os.path.join(root, f), os.path.join(root, "{}{}".format(prefix, f)))  
    #% Bring all files to bas folder (dest_dir)
    for root, dirs, files in os.walk(root_dir):  # replace the . with your starting directory
       for file in files:
           path_file = os.path.join(root,file)
           shutil.copy2(path_file,dest_dir) # change you destination dir
    #https://stackoverflow.com/questions/23196512/how-to-copy-all-files-from-a-folder-including-sub-folder-while-not-copying-the/23197213
    #https://camerondwyer.com/2013/07/05/how-to-copy-an-entire-folder-structure-without-copying-the-files-tip-for-starting-the-new-financial-year
    #% Copy only (ok 0,1,2,.. ending) *csv files to correct folder (ub, np, etc.)
    copyfiles(dest_dir, dest_final, "*.csv")  
    directory = dest_dir
    test = os.listdir( directory ) 
    for item in test:
        if item.endswith(".cc3d"):
            os.remove( os.path.join( directory, item ) )
        if item.endswith(".vtk"):
            os.remove( os.path.join( directory, item ) )  
        if item.endswith(".dml"):
            os.remove( os.path.join( directory, item ) )
        if item.endswith(".pyc"):
            os.remove( os.path.join( directory, item ) )  
        if item.endswith(".piff"):
            os.remove( os.path.join( directory, item ) )
        if item.endswith(".py"):
            os.remove( os.path.join( directory, item ) )  
        if item.endswith(".xml"):
            os.remove( os.path.join( directory, item ) ) 
        if item.endswith(".png"):
            os.remove( os.path.join( directory, item ) ) 
#%%Do the removing, i.e. the actual operation of the previous function:
renmov(root_dir = 'C://Users//pauli//CC3DWorkspace//NewSimulation_ParameterScan_upnpadh_2diii//',
           dest_dir = 'C:/Model data/i/',
                dest_final = 'C://Model data//2d7t//NewSimulation_ParameterScan_upnpadh_2d_v2nu//')     
#%%Deleting files
#https://python-forum.io/Thread-Delete-files-inside-Folder-and-subfolders
os.chdir("C:\\Model data\\2D_rnd\\")
for root, dirs, files in os.walk(".", topdown = False):
   for file in files:
      print(os.path.join(root, file))
      os.remove(os.path.join(root, file)) 
inputpath = 'C://Users/pauli/CC3DWorkspace/2drndfield_nrm//'
outputpath = 'C://python//Model_results//2drndfield_nrm//' 
#careful with the names.. and endigs and /   
for dirpath, dirnames, filenames in os.walk(inputpath):
    structure = os.path.join(outputpath, dirpath[len(inputpath):])
    if not os.path.isdir(structure):
        os.mkdir(structure)
    else:
        print("Folder does already exits!")          
