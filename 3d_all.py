# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:15:22 2019

@author: pauli
"""

#3D ANALYSIS codes and routines, tikka 7.8.19, 8 phases:
# 1. import packages, 2. import model data, 3. peforming the calculations with functions,
# 4-7. create the functions for the calculations, and 8. create a good fucntion for data skimming 
# and importing from your folder systems

#%% 1) IMPORT PACKAGES
import pandas as pd #for importing files
# https://pandas.pydata.org/pandas-docs/version/0.18.1/generated/pandas.DataFrame.html
import numpy as np  #for calculations, array manipulations, and fun :)
import matplotlib.pyplot as plt #for scientifical plots
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
import glob
import sys
import fnmatch 
from os.path import join, isfile
import shutil

#%% SEE THE 'IMPORT OF MODEL FILES FROM WORKSPACE FOLDER' -CODE BELOW
#e.g. dest_final = 'C://Model data//3DRND//Opti//.../') 
# https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory

#%% 2)IMPORT MODEL DATA
#Example locations for results:
#C:\python\Model_results\2D_new\OptiRand
#% Kaikkien tiedostojen tuonti, ok (x): 
# x adh3d, x np_adh3d, x np3d, x ref3d, x ub_adh3d, x ub3d, (ub_npadh3d)
#        C:\Model data\fields\NewSimulation_ParameterScan_np_adh3d_OptiRand
#C://Model data//3drndfields_norm//np2//
#['C:/Model data/3drndfields_norm\\NewSimulation_ParameterScan_npadh_field_r',
d = 'C:/Model data/3drndfields_norm'
[os.path.join(d, o) for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]
directory="C://Model data//3drndfields_pso//np2//*distances.csv"
dataframes = []
all_files2=(glob.glob(directory))
# Create the list for the three DataFrames you want to create:
for filename in all_files2:
    dataframes.append(pd.read_csv(filename))
#% Scaling to experimental frame (Combes et al. 2016)
df_Combes = pd.read_excel('C:/python/' + 'mmc4.xlsx')
#% In case column names are misplaces (during calculations)
for i in range(len(dataframes)):
    dataframes[i].rename(columns={'        z_position_(px)':'z_position_(px)'},inplace=True)

#%% 3) PEFORM THE CALCULATIONS TO DATA WITH THE BELOW FUCNTIONS:
#SPEEDS    
#Speeds, the time is (max mcs+(zero values if applicalbe))/10
yesx=[]
yesx2=[]
yesx3=[]
yesx=Speed_3D_ci(dataframes,cond='NPCells',conda='rnd',time=101,dt=10,name='C:/python/Model_results/speedCI_rnd_3D_NP.csv')#to combes
yesx2=Speed_3D_ci(dataframes,cond='ACells',conda='rnd',time=101,dt=10,name='C:/python/Model_results/speedCI_rnd_3D_MM.csv')
yesx3=Speed_3D_ci(dataframes,cond='both',conda='rnd',time=101,dt=10,name='C:/python/Model_results/speedCI_rnd_3D_NP&MM.csv')#to oulu
resulti = pd.concat([yesx, yesx2, yesx3], axis=1, sort=False)
resulti.to_csv('C:/python/Model_results/speedCI_rnd_3D_all_tikka25919.csv',index=False,header='infer')
#%Distances
yes2=[]
yes3=[]
yes4=[]
yes2=Dist_3D_ci(dataframes,cond='ACells',conda='rnd',time=101,name='C:/python/Model_results/distCI_rnd_3D_MM.csv')
yes3=Dist_3D_ci(dataframes,cond='NPCells',conda='rnd',time=101,name='C:/python/Model_results/distCI_rnd_3D_NP.csv')
yes4=Dist_3D_ci(dataframes,cond='both', conda='rnd',time=101,name='C:/python/Model_results/distCI_rnd_3D_NP&MM.csv')
resultia = pd.concat([yes2, yes3, yes4], axis=1, sort=False)
resultia.to_csv('C:/python/Model_results/distCI_rnd_3D_all_tikka25919.csv',index=False,header='infer')    
#%Concentrations: (separate run?)
yesxi=[]    
yesxi2=[]
yesxi3=[]
yesxi=conc_3D_ci(dataframes,cond='NPCells',conda='rnd',time=101,name='C:/python/Model_results/concCI_rnd_3D_NP.csv')
yesxi2=conc_3D_ci(dataframes,cond='ACells',conda='rnd',time=101,name='C:/python/Model_results/concCI_rnd_3D_MM.csv')
yesxi3=conc_3D_ci(dataframes,cond='both',conda='rnd',time=101,name='C:/python/Model_results/concCI_rnd_3D_NP&MM.csv')
resultib = pd.concat([yesxi, yesxi2, yesxi3], axis=1, sort=False)
resultib.to_csv('C:/python/Model_results/concCI_rnd_3D_all_tikka25919.csv',index=False,header='infer') 
#% Amounts:
resulti= Amount_3D_ci(dataframes,
                      cond='NPCells',cond1='tip', conda='pers2',time=101,name='C:/python/Model_results/UB_NPtip_amount_24919tikka'
                      )    #np_adh3d
resulti= Amount_3D_ci(dataframes,
                      cond='ACells',cond1='tip', conda='pers2',time=101,name='C:/python/Model_results/UB_MMtip_amount_24919tikka'
                      )    #np_adh3d
resulti= Amount_3D_ci(dataframes,
                      cond='NPCells',cond1='corner', conda='pers2',time=101,name='C:/python/Model_results/UB_NPcorner_amount_24919tikka'
                      )    #np_adh3d
resulti= Amount_3D_ci(dataframes,   
                      cond='ACells',cond1='corner', conda='pers2',time=101,name='C:/python/Model_results/UB_MMcorner_amount_24919tikka'
                      )    #np_adh3d

#%% 4) CREATE THE FUNCTIONS FOR CALCULATING SPEEDS (LONG DESTCTIPTIONSTCTIPTION)  
#First function for speeds
#Select criteria; half of the cells
def crit_fun(dataframes,x0,x1,y0,y1,z0,z1):
    dataframes2 = []
#    crit0=dataframes['time_mcs']==0
    crit1=dataframes['x_position_(px)']>=x0
    crit2=dataframes['y_position_(px)']>=y0
    crit3=dataframes['z_position_(px)']>=z0
    crit4=dataframes['x_position_(px)']<=x1
    crit5=dataframes['y_position_(px)']<=y1
    crit6=dataframes['z_position_(px)']<=z1    
    criteria_all=crit1 & crit2 & crit3 & crit4 & crit5 & crit6    
    dataframes2=dataframes[criteria_all] #np half (thus h in the name df_dis_nph) 
    return dataframes2    
 
#Make an auxialiry function for next criteria speed (and for other similarly) function:
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3   
 
#%    Valitaan oikeat solut listana, (myöhemmin matriisina, josta CI lasku)
def speed3_preli(dataframes,cond,conda):
    df_dist_adh3d=dataframes
# Muokkaus
# Valitse yhden tiedoston kaikki NP solut yhtena ajan kohtana:
# Solujen nimet:
    if conda == 'rnd':       
        dfcell_names3d=[]    
        dfcell_names3d=df_dist_adh3d   
        dfcell_names3d=dfcell_names3d.ix[0:391,0:2] #in 3D there are more cells than in 2D
        dfcell_names3d.ix[:,1]=list(np.where(dfcell_names3d.ix[:,1] > 1, 'ACells', 'NPCells'))
        dfcell_names3da=dfcell_names3d.ix[:,0:2] 
    elif conda == 'norm':
        dfcell_names3d= pd.read_csv('all cells uniform3d_v2.csv', delimiter=";", header=None)
        dfcell_names3da=dfcell_names3d[[0,1]]            
#% selecting common indeces in the list of lists of indeces 
    dfoki22=[]
    setti=[]
    settit=[]
    time=[]
#    cond='ACells'
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
    #%#%    elif cond == 'both':  
    #39,61, ok ->104 cells?? more likely.. 196 no, it is 104, since around half amount of cells
    settit=list(crit_fun(dfoki22,0,66,0,81,24,76).index.unique())
    #%    if cond == 'corner':
    setti2=intersection(setti, settit)  
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
        setti22=list(a) 
#%  elif cond == 'tip':
    c3=(40 <= dfoki22['y_position_(px)'])
    dataframesii=pd.DataFrame()
    dataframesii=dfoki22[c3]
    b=list(dataframesii.index.unique())
#    settiii=list(setti2)
    setti3=intersection(settii,b) #ok
    if setti3 ==[]:
        setti3=list(b)
    return settit,setti22, setti3 #both, corner, tip
#%Test
setti, setti2, setti3  = speed3_preli(dataframes[2],cond='NPCells', conda='rnd')    #both,corner, tip

#%xyz dis for NPCells, ACells, and Both
def x_y_z_dis_fun(dataframes,setti,time):
    from random import randrange, uniform #just in case here, needed for random indexing    
#%The X values:
# Append model values from file with your indeces (with 'for loop') to obtain a list of lists
# https://stackoverflow.com/questions/522563/accessing-the-index-in-python-for-loops
    #%df_model
    dfJEE=dataframes.set_index("cell_no")
    index = 0
    mmf=[]
    for index in setti: 
        mmf.append(np.array(dfJEE.ix[index,'x_position_(px)'])) # mmf[0][-1]: Out[54]: 22.818181818200003 
#    % Removing 'nans'
    for i in range(len(mmf)):
        mmf[i][np.where(np.isnan(mmf[i]))]=+0.1
        for j in range(len(mmf[i])):
            if mmf[i][j] == +0.1:
                lenin=int(round(len(mmf[i])/4)) # latest previous values
                mmf[i][j] == abs(mean(mmf[i][-lenin:-2])) # replacing with mean of latest previous values
    t_jee=time      
# The symmetrical numpy matrix made from this list of list by padding constant values to the end 
# of those lists that are below the amount of cells 
    mtt=np.zeros((len(setti),t_jee))
    mmf=np.array(mmf)
#    https://cognitiveclass.ai/blog/nested-lists-multidimensional-numpy-arrays
    if mtt.shape == mmf.shape:
        mtt=mmf
    elif mtt.shape != mmf.shape:
        for i in range(len(mmf)):
            Ashape=mmf[i].shape[0]
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
# https://stackoverflow.com/questions/3996904/generate-random-integers-between-0-and-9      
#% The Y values extracted   
    index = 0
    mmf2=[]
#   Instead of loop, use while, and append to obtain a list of lists
#   https://stackoverflow.com/questions/522563/accessing-the-index-in-python-for-loops
    for index in setti:
        mmf2.append(np.array(dfJEE.ix[index,'y_position_(px)'])) 
    # Removing 'nans'
    for i in range(len(mmf2)):
        mmf2[i][np.where(np.isnan(mmf2[i]))]=+0.1
        for j in range(len(mmf2[i])):
            if mmf2[i][j] == +0.1:
                lenin3=int(round(len(mmf2[i])/4))
                mmf2[i][j] == abs(mean(mmf2[i][-lenin3:-2])) 
                # excluding the 0.1 with '-2' index
                # https://stackoverflow.com/questions/5124376/convert-nan-value-to-zero
# y values are in the mtt2 matrix
    from random import randrange, uniform
    mtt2=np.zeros((len(setti),t_jee))
    mmf2=np.array(mmf2)
#    https://cognitiveclass.ai/blog/nested-lists-multidimensional-numpy-arrays
    if mtt2.shape == mmf2.shape:
        mtt2=mmf2
    elif mtt2.shape != mmf2.shape:
        for i in range(len(mmf2)):
            Ashape=mmf2[i].shape[0]
            new_shape=t_jee 
            shape_diff = np.asarray(new_shape) - np.asarray(Ashape)
            mtt2[i,:]=np.lib.pad(mmf2[i], (0,shape_diff), 'constant', constant_values=(0.1))
# changing those constant values (0.1) to the random values close to the end values
    uli3=mtt2.shape[0]
    uli4=mtt2.shape[1]
    lenin=[]
    for i in range(uli3):
        for j in range(uli4):
            if mtt2[i,j]==0.1:
                lenin4=int(round(len(mmf2[i])/4)) # final std range (the fourth quartile of last values)
                mtt2[i,j]=mmf2[i][-1]+uniform(-1,1)*std(mmf2[i][-lenin4:-1])*0.25
# The y values should not go below 1.5 (too close to axis!!)
    for i in range(uli3):
        for j in range(uli4):
            if mtt2[i,j]<1.5:
                mtt2[i,j]=mean(mtt2[i,0:j])+std(mtt2[i,0:j])*0.25*uniform(0,1)
# The Z values extracted
    index = 0
    mmf3=[]
    for index in setti:
        mmf3.append(np.array(dfJEE.ix[index,'z_position_(px)']))        
    # Removing 'nans'
    for i in range(len(mmf3)):
        mmf3[i][np.where(np.isnan(mmf3[i]))]=+0.1
        for j in range(len(mmf3[i])):
            if mmf3[i][j] == +0.1:
                lenin5=int(round(len(mmf3[i])/4))
                mmf3[i][j] == abs(mean(mmf3[i][-lenin5:-2])) 
                # excluding the 0.1 with '-2' index
                # https://stackoverflow.com/questions/5124376/convert-nan-value-to-zero              
# z values are in the mtt3 matrix
    mtt3=np.zeros((len(setti),t_jee))
    mmf3=np.array(mmf3)
#    https://cognitiveclass.ai/blog/nested-lists-multidimensional-numpy-arrays
    if mtt3.shape == mmf3.shape:
        mtt3=mmf3
    elif mtt3.shape != mmf3.shape:
        for i in range(len(mmf3)):
            Ashape=mmf3[i].shape[0]
            new_shape=t_jee 
            shape_diff = np.asarray(new_shape) - np.asarray(Ashape)
            mtt3[i,:]=np.lib.pad(mmf3[i], (0,shape_diff), 'constant', constant_values=(0.1))
# changing those constant values (0.1) to the random values close to the end values  
    uli5=mtt3.shape[0]
    uli6=mtt3.shape[1]
    for i in range(uli5):
        for j in range(uli6):
            if mtt3[i,j]==0.1:
                lenin6=int(round(len(mmf2[i])/4)) # final std range (the fourth quartile of last values)
                mtt3[i,j]=mmf3[i][-1]+uniform(-1,1)*std(mmf3[i][-lenin6:-1])*0.25
# The z values should not go below 1.5 (too close to axis!!)
    for i in range(uli5):
        for j in range(uli6):
            if mtt3[i,j]<1.5:
                mtt3[i,j]=mean(mtt3[i,0:j])+std(mtt3[i,0:j])*0.25*uniform(0,1)
    return mtt, mtt2, mtt3
#% Test is always a good thing, also here in the functions sections:
# mtt, mtt2, mtt3=x_y_z_dis_fun(dataframes[0],setti,time=101)

#% New speed function
def news1(dataframes,setti,time,dt):
#%
    axan, byan, czan = x_y_z_dis_fun(dataframes,setti,time)
    #Selecting the correct spaces
    index_cells=len(setti)            # max cell values
    index_time=time
    r=np.zeros((index_cells,index_time)) # here should be the size of the matrix len(...)
    for j in range(index_time-1):
        r[:,j] = np.sqrt((axan[:,(j+1)]-axan[:,j])**2+(byan[:,(j+1)]-byan[:,j])**2+(czan[:,(j+1)]-czan[:,j])**2)
    r2=r/dt
    r2=r2[:,0:time]   
    return r2
#Test:
#diffa3=[]
#diffa3=news1(dataframes[2],setti,time=100,dt=10)    

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
#tip
    speed=news1(dataframes,setti3,time,dt)  
    yes=pd.DataFrame(speed)    #tip
    yes2=pd.DataFrame(speed2) #corner
    yes3=pd.DataFrame(speed3) #both        
  #%              
    return yes, yes2, yes3 #tip,corner,both speeds
#yes, yes2, yes3=speed_fun(dataframes[0],setti,setti2,setti3,time=101,dt=10)

#% Calculating CI, also for other measures, such as distances laskeminen
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
    yes=[]
    yes2=[]
    yes3=[]
    for i in range(time):
        yes.append(mean_confidence_interval(result[[i]], confidence=0.95))
        yes2.append(mean_confidence_interval(result2[[i]], confidence=0.95))
        yes3.append(mean_confidence_interval(result3[[i]], confidence=0.95))     
#    #eli taa on yhdelle riville... tais olla siina.. :)
    yes=pd.DataFrame(yes)
    yes2=pd.DataFrame(yes2) 
    yes3=pd.DataFrame(yes3)   
    for i in range(time):
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
    yes3[3]=tuple(range(time))
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

#%% 5) CREATE THE FUNCTIONS FOR CALCULATING DISTANCES (LONG DESCRIPTION)  
#First function for distances
#Select criteria; half of the cells
def crit_fun(dataframes,x0,x1,y0,y1,z0,z1):
    dataframes2 = []
#    crit0=dataframes['time_mcs']==0
    crit1=dataframes['x_position_(px)']>=x0
    crit2=dataframes['y_position_(px)']>=y0
    crit3=dataframes['z_position_(px)']>=z0
    crit4=dataframes['x_position_(px)']<=x1
    crit5=dataframes['y_position_(px)']<=y1
    crit6=dataframes['z_position_(px)']<=z1    
    criteria_all=crit1 & crit2 & crit3 & crit4 & crit5 & crit6    
    dataframes2=dataframes[criteria_all] #np half (thus h in the name df_dis_nph) 
    return dataframes2    

#%  3D distance arvot listana, (myöhemmin matriisina, josta CI lasku)
def dist3_preli(dataframes,cond,conda):
    df_dist_adh3d=dataframes    
#% Muokkaus
#% Valitse yhden tiedoston kaikki NP solut yhtena ajan kohtana:
# Solujen nimet:
    if conda == 'rnd':       
        dfcell_names3d=[]    
        dfcell_names3d=df_dist_adh3d   
        dfcell_names3d=dfcell_names3d.ix[0:391,0:2]
        dfcell_names3d.ix[:,1]=list(np.where(dfcell_names3d.ix[:,1] > 1, 'ACells', 'NPCells'))
        dfcell_names3da=dfcell_names3d.ix[:,0:2] 
    elif conda == 'norm':
        dfcell_names3d= pd.read_csv('all cells uniform3d_v2.csv', delimiter=";", header=None)
        dfcell_names3da=dfcell_names3d[[0,1]]            
# selecting common indeces in the list of lists of indeces 
    dfoki22=[]
    setti=[]
    time=[]
#    cond='both'
    if cond == 'both':           
        dfcell_names3da.columns=['a','b']
        c1=(dfcell_names3da['b'] == 'NPCells')
        c2=(dfcell_names3da['b'] == 'ACells')
        cond=c1+c2
        setti=dfcell_names3da.loc[cond,"a"] #all
    else:
        setti=dfcell_names3da[dfcell_names3da.ix[:,1]==cond].ix[:,0]
    dfoki22=df_dist_adh3d.set_index("cell_no")
    dfoki22=dfoki22.loc[list(setti),:]
    time=len((dfoki22.set_index("time_mcs")).index.unique()) #length
    dfoki22=dfoki22.loc[list(crit_fun(dfoki22,0,56,0,81,24,76).index.unique()),:] #39,61, ok ->104 cells
    settia=list(crit_fun(dfoki22,0,56,0,81,24,76).index.unique()) 
    return dfcell_names3da,settia,time
#Test is always good:
#dfcell_names3da,settia,time=dist3_preli(dataframes[0],cond='NPCells',conda='rnd')

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
#    elif cond == 'tip':
    c2=(dataframes['y_position_(px)'] >= 40)
    dataframesii=dataframes[c2]
    dataframesii=dataframesii.set_index("cell_no")
    b=list(dataframesii.index.unique())
    setti3=intersection(b, settia) #ok             
    index=0
    mmf=[]
    mmf2=[]  
    for index in setti2:
        mmf.append(np.array(dataframesi.loc[index,'cell_dist_to_corner'])) 
    index=0
    for index in setti3:    
        mmf2.append(np.array(dataframesii.loc[index,'cell_dist_to_tip'])) #tip
#        https://codereview.stackexchange.com/questions/156447/subtract-multiple-columns-in-pandas-dataframe-by-a-series-single-column
# The symmetrical numpy matrix made from this list of list by padding constant values to the end 
# of those lists that are below the amount of cells 
    mtt=np.zeros((len(mmf),time)) #test also: len(dataframes[0].ix[:,1].unique())
    mt2=np.zeros((len(mmf2),time)) #test also: len(dataframes[0].ix[:,1].unique())
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
   #3D version
    array = dfcell_names3da.ix[:,0]
    item = list(setti2)
    lista=[]
    for i in range(len(item)):
        lista.append((np.where(array.ix[0:]==item[i]))[0][0])
    row_names=dfcell_names3da.ix[lista,1]
    row_names=list(row_names) #row was one...   
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
#Test:
#mtt, mt2 = dishi_fun(dataframes[0],dfcell_names3da,settia,time,df_Combes)

#% 3D distance CI:s    
def Dist_3D_ci(dataframes,cond,conda,time,name):
    tot  = []
    tot2 = []
    for i in range(len(dataframes)):
        tot.append(dist3_preli(dataframes[i],cond, conda))
        tot2.append(dishi_fun(dataframes[i],tot[i][0],tot[i][1],time,df_Combes)) #ub half.. %
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
    for i in range(time): #check if 100 (0-990) or 101 (0-1000)
        yes.append(mean_confidence_interval(result[[i]], confidence=0.95))
        yes2.append(mean_confidence_interval(result2[[i]], confidence=0.95))        
    yes=pd.DataFrame(yes)
    yes2=pd.DataFrame(yes2)      
    for i in range(time):
        yes.loc[i,1]=float(yes.loc[i,1])
        yes.loc[i,2]=float(yes.loc[i,2])
        yes2.loc[i,1]=float(yes2.loc[i,1])
        yes2.loc[i,2]=float(yes2.loc[i,2])
#or https://stackoverflow.com/questions/44603615/plot-95-confidence-interval-errorbar-python-pandas-dataframes)
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

#%% 6) CREATE THE FUNCTIONS FOR CALCULATING CONCETRATIONS (LONG DESCRIPTION)  
#First function for CONCENTRATIONS
#Select criteria; half of the cells
def crit_fun(dataframes,x0,x1,y0,y1,z0,z1):
    conc = []
#    crit0=dataframes['time_mcs']==0
    crit1=dataframes['x_position_(px)']>=x0
    crit2=dataframes['y_position_(px)']>=y0
    crit3=dataframes['z_position_(px)']>=z0
    crit4=dataframes['x_position_(px)']<=x1
    crit5=dataframes['y_position_(px)']<=y1
    crit6=dataframes['z_position_(px)']<=z1    
    criteria_all=crit1 & crit2 & crit3 & crit4 & crit5 & crit6    
    dataframes2=dataframes[criteria_all] #np half (thus h in the name df_dis_nph) 
    return dataframes2    
#%   Valitaan oikeat solut listana, (myöhemmin matriisina, josta CI lasku)
def conc3_preli(dataframes,cond, conda):
    df_dist_adh3d=dataframes
#% 2) Muokkaus
#% Valitse yhden tiedoston kaikki NP solut yhtena ajan kohtana:
# Solujen nimet:
    if conda == 'rnd':       
        dfcell_names3d=[]    
        dfcell_names3d=df_dist_adh3d   
        dfcell_names3d=dfcell_names3d.ix[0:391,0:2]
        dfcell_names3d.ix[:,1]=list(np.where(dfcell_names3d.ix[:,1] > 1, 'ACells', 'NPCells'))
        dfcell_names3da=dfcell_names3d.ix[:,0:2] 
    elif conda == 'norm':
        dfcell_names3d= pd.read_csv('all cells uniform3d_v2.csv', delimiter=";", header=None)
        dfcell_names3da=dfcell_names3d[[0,1]]            
#% selecting common indeces in the list of lists of indeces 
    dfoki22=[]
    setti=[]
    settit=[]
    time=[]
#    cond='ACells'
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
    #%#%    elif cond == 'both':  
#    dfoki22=dfoki22.loc[list(crit_fun(dfoki22,0,66,0,81,24,76).index.unique()),:] 
    #39,61, ok ->104 cells?? more likely.. 196 no, it is 104, since around half amount of cells
    settit=list(crit_fun(dfoki22,0,66,0,81,24,76).index.unique())
    #%    if cond == 'corner':
    setti2=intersection(setti, settit)
    dataframesa=pd.DataFrame()
    dataframesa=dfoki22
    c1=(40 >= dataframesa['y_position_(px)'])
    dataframesi=dataframesa[c1]
#    c2=(65 > dataframesi['x_position_(px)'])    
#    dataframesix=dataframesi[c2]
    a=list(dataframesi.index.unique())
    settii=list(setti2) #onko tama setti eri, kuin alla?
    setti22=intersection(settii, a) #ok
#%  elif cond == 'tip':
    c3=(40 <= dfoki22['y_position_(px)'])
    dataframesii=pd.DataFrame()
    dataframesii=dfoki22[c3]
    b=list(dataframesii.index.unique())
    setti3=intersection(settii,b) #ok
    return settit,setti22, setti3 #both, corner, tip
# Test:
#setti, setti2, setti3  = conc3_preli(dataframes[2],cond='ACells', conda='rnd')    #both,corner, tip

#%xyz dis for NPCells, ACells, and Both
def x_y_z_dis_fun2(dataframes,setti,time):
    from random import randrange, uniform #just in case here, needed for random indexing    
#%Te X values:
# https://stackoverflow.com/questions/522563/accessing-the-index-in-python-for-loops
    dfJEE=dataframes.set_index("cell_no")
    index = 0
    mmf=[]
    for index in setti: 
        mmf.append(np.array(dfJEE.ix[index,'cell_field'])) # mmf[0][-1]: Out[54]: 22.818181818200003
#    % Removing 'nans'
    for i in range(len(mmf)):
        mmf[i][np.where(np.isnan(mmf[i]))]=+0.1
        #
        for j in range(len(mmf[i])):
            if mmf[i][j] == +0.1:
                lenin=int(round(len(mmf[i])/4)) # latest previous values
                mmf[i][j] == abs(mean(mmf[i][-lenin:-2])) # replacing with mean of latest previous values
    t_jee=time    
# The symmetrical numpy matrix made from this list of list by padding constant values to the end 
# of those lists that are below the amount of cells 
    mtt=np.zeros((len(setti),t_jee))
    for i in range(len(mmf)):
        Ashape=mmf[i].shape[0]
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
#Test:
#mtt=x_y_z_dis_fun2(dataframes[0],setti,time)
#
#% New concentration function
def news(dataframes,setti,time):
#%
    axan= x_y_z_dis_fun2(dataframes,setti,time)
    #Selecting the correct spaces
    index_cells=len(setti)            # max cell values
    index_time=time
    r=np.zeros((index_cells,index_time)) # here should be the size of the matrix len(...)
    for j in range(index_time-1):
        r[:,j] = abs((axan[:,(j+1)]-axan[:,j]))
    r2=r[:,0:(time)]   
    return r2
#Test:
#diffa3=[]
#diffa3=news(dataframes[2],setti,time)    
#plt.plot(np.mean(diffa3, axis=0))

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
#setti, setti2, setti3  = speed3_preli(dataframes[3],cond='NPCells')    #both,corner, tip    
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
    for i in range(len(dataframes)): #check separately
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
    for i in range(time): #check times..
        yes.append(mean_confidence_interval(result[[i]], confidence=0.95))
        yes2.append(mean_confidence_interval(result2[[i]], confidence=0.95))
        yes3.append(mean_confidence_interval(result3[[i]], confidence=0.95))
    yes=pd.DataFrame(yes)
    yes2=pd.DataFrame(yes2) 
    yes3=pd.DataFrame(yes3)   
    for i in range(time):
        yes.loc[i,1]=float(yes.loc[i,1])
        yes.loc[i,2]=float(yes.loc[i,2])
        yes2.loc[i,1]=float(yes2.loc[i,1])
        yes2.loc[i,2]=float(yes2.loc[i,2])
        yes3.loc[i,1]=float(yes3.loc[i,1])
        yes3.loc[i,2]=float(yes3.loc[i,2])
##or https://stackoverflow.com/questions/44603615/plot-95-confidence-interval-errorbar-python-pandas-dataframes
##    result = pd.concat([df1, df4], axis=1, sort=False)
##% Kääntö
    yes[1] = yes[1].astype(float)
    yes[2] = yes[2].astype(float) 
    yes2[1] = yes2[1].astype(float)
    yes2[2] = yes2[2].astype(float) 
    yes3[1] = yes3[1].astype(float)
    yes3[2] = yes3[2].astype(float) 
    yes3[3]=tuple(range(time))
    resulti = pd.concat([yes, yes2, yes3], axis=1, sort=False)
#https://datatofish.com/convert-string-to-float-dataframe/ 
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

#%% 7) CREATE THE FUNCTIONS FOR CALCULATING CELL QAUNTITIES OR AMOUNTS (LONG DESCRIPTION)  
#%%Amounts:
##https://rohitmidha23.github.io/Matplotlib-Explained/
##https://matplotlib.org/tutorials/introductory/pyplot.html
##https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.xticks.html

#%First function concentrations, the half criteria:
def crit_funt(dataframes,y0, cond1):
    dataframes2 = []
    if cond1 == 'tip': 
        crit2=dataframes['y_position_(px)']>=y0 #'corner' or 'tip, above 40'
    elif cond1 == 'corner': 
        crit2=dataframes['y_position_(px)']<=y0 #'corner', below 40 or 'tip, above 40'
    dataframes2=dataframes[crit2]
    return dataframes2 
#Test:
#dataframes2=crit_funt(dataframes[0],40,cond1='corner')

#The cell amount function in 3d:
def amount(dataframes, cond, cond1, conda):
    dataframes2=crit_funt(dataframes,40,cond1)
    df_dist_adh3d=dataframes2
#    conda='pers2'
    if conda == 'pers':    #check if ok..       
        dfcell_names3d= pd.read_csv('all cells uniform3d_v3a.csv', delimiter="\t", header=None)
        dfcell_names3da=dfcell_names3d[[0,1]]
    elif conda == 'no-pers':
        dfcell_names3d= pd.read_csv('all cells uniform3d_v2.csv', delimiter=";", header=None)
        dfcell_names3da=dfcell_names3d[[0,1]]
    elif conda == 'pers2':       
        dfcell_names3d=[]    
        dfcell_names3d=dataframes 
        dfcell_names3d=dfcell_names3d.ix[0:391,0:2]
        dfcell_names3d.ix[:,1]=list(np.where(dfcell_names3d.ix[:,1] > 1, 'ACells', 'NPCells'))
        dfcell_names3da=dfcell_names3d.ix[:,0:2] 
#%    https://stackoverflow.com/questions/27674880/python-replace-a-number-in-array-with-a-string
    dfoki22=[]
    setti=[]
    settit=[]
    time=[]
#    cond == 'NPCells'
    if cond == 'NPCells':
        setti=dfcell_names3da[dfcell_names3da.ix[:,1]==cond].ix[:,0]
    elif cond == 'ACells':
        setti=dfcell_names3da[dfcell_names3da.ix[:,1]==cond].ix[:,0]   
    dfoki22=df_dist_adh3d.set_index("cell_no")
    dfoki22=dfoki22.loc[list(setti),:]
    dfoki22=dfoki22.dropna()
#    https://stackoverflow.com/questions/2161752/how-to-count-the-frequency-of-the-elements-in-a-list/2162045
#    https://stackoverflow.com/questions/41217310/get-index-of-a-row-of-a-pandas-dataframe-as-an-integer/42853445
    a=dfoki22['time_mcs']
    from itertools import groupby
    results = {value: len(list(freq)) for value, freq in groupby(sorted(a))}
    return results
#% Test the function: 
#results =amount(dataframes[22], cond = 'NPCells', cond1='corner', conda='pers2') 
#plt.plot(results.values())

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
#% 4) Kääntö
    yes[1] = yes[1].astype(float)
    yes[2] = yes[2].astype(float) 
    resulti = pd.concat([yes], axis=1, sort=False)
#https://datatofish.com/convert-string-to-float-dataframe/ 
    resulti.columns = ['Cells_region_avg', 'Cells_region_CI_min', 'Cells_region_CI_max']               
#https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas
    resulti.to_csv(name, index=False, header='infer') #Tallennus samassa    
    return resulti   

#%% 8) BRING OR CURATE FILES FROM OTHER FOLDERS TO ANTOHER INORDER TO GET THEM IN OK SHAPE FOR CALCULATIONS:
#%%Rename (and move model simulation) files (yes, three phases to get the from A to B)
#https://stackoverflow.com/questions/38948774/python-append-folder-name-to-filenames-in-all-sub-folders
#Auxialiry funciton for next 'remove' function:    
def copyfiles(srcdir, dstdir, filepattern):
        def failed(exc):
            raise exc
    
        for dirpath, dirs, files in os.walk(srcdir, topdown=True, onerror=failed):
            for file in fnmatch.filter(files, filepattern):
                shutil.move(os.path.join(dirpath, file), dstdir)
            break # no recursion	
#Remove and drive the folders to the right folder function:			
def renmov(root_dir = 'C://Users//pauli//CC3DWorkspace//3drnd_o//ppp',
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
    #       if files.endswith(".csv"):
           path_file = os.path.join(root,file)
           shutil.copy2(path_file,dest_dir) # change you destination dir
    #https://stackoverflow.com/questions/23196512/how-to-copy-all-files-from-a-folder-including-sub-folder-while-not-copying-the/23197213
    #https://camerondwyer.com/2013/07/05/how-to-copy-an-entire-folder-structure-without-copying-the-files-tip-for-starting-the-new-financial-year/
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

#%% Copy folder structure
#https://stackoverflow.com/questions/40828450/how-to-copy-folder-structure-under-another-directory      
#inputpath = 'C:/python/Model_results/3DRND/Norm/'
#outputpath = 'C:/python/Model_results/3DRND/Opti/'
#C:\Users\pauli\CC3DWorkspace\3drnd_field
#C:\python\Model_results\3Drnd_field_new
#C:\Users\pauli\CC3DWorkspace\3d_rnd_field_opti   
#C:\Model data\3drndfields_pso
inputpath = 'C:/Users/pauli/CC3DWorkspace/3d_rnd_field_opti/'
outputpath = 'C:/Model data/3drndfields_pso/' #careful with the names.. and endigs and / 
for dirpath, dirnames, filenames in os.walk(inputpath):
    structure = os.path.join(outputpath, dirpath[len(inputpath):])
    if not os.path.isdir(structure):
        os.mkdir(structure)
    else:
        print("Folder does already exits!")             

#%% Get the names of folder paths:
#ax=os.walk(outputpath)   
#        https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
d = 'C://Users//pauli//CC3DWorkspace//3d_rnd_field_opti//'
[os.path.join(d, o) for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]           

#%% Drive the files to right folder:    
#(check above)         
 #['C://Users//pauli//CC3DWorkspace//3d_rnd_field_opti//NewSimulation_ParameterScan_npadh_field_or',
# 'C://Users//pauli//CC3DWorkspace//3d_rnd_field_opti//NewSimulation_ParameterScan_npadh_field_or2',
# 'C://Users//pauli//CC3DWorkspace//3d_rnd_field_opti//NewSimulation_ParameterScan_np_field_or2',
# 'C://Users//pauli//CC3DWorkspace//3d_rnd_field_opti//NewSimulation_ParameterScan_ubnpadh2575_field_or22',
# 'C://Users//pauli//CC3DWorkspace//3d_rnd_field_opti//NewSimulation_ParameterScan_ubnpadh_field_or',
# 'C://Users//pauli//CC3DWorkspace//3d_rnd_field_opti//NewSimulation_ParameterScan_ub_field_or2']    
#C:\Users\pauli\CC3DWorkspace\3d_rnd_field_opti   
#C:\Model data\3drndfields_pso
# C:\Users\pauli\CC3DWorkspace\3drnd_field\NewSimulation_ParameterScan_np_field_r
# C:\Users\pauli\CC3DWorkspace\3d_rnd_field_opti\NewSimulation_ParameterScan_np_field_or2
renmov(root_dir = 'C://Users//pauli//CC3DWorkspace//3d_rnd_field_opti//NewSimulation_ParameterScan_np_field_or2//',
           dest_dir = 'C:/Model data/i/',
                dest_final = 'C://Model data//3drndfields_pso//np2//')     

   
