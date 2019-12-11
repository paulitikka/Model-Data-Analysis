# -*- coding: utf-8 -*-i
"""
Created on Mon Nov 19 11:22:12 2018

@author: pauli
"""

#SOM ANALYSIS codes and routines, tikka 7.8.19, 10 phases:
# 1. import packages, 2. import model data, 3. converst data suitable for som, dsom matrix (i.e. right scope and .csv),
# 4. check the speed regions manually from dsom, 5. excecute and plot SOM nodes, 6. group the nodes, 
# 7. choose and show the best node groups
# 8. semisom function, 9. execute the selection of best groups, semisom, and speed coordinate plots, 
# 10. save essential data for reanalysis
 
#%% 1) IMPORT PACKAGES
import aok_tikka71018 #own pacakes, may be needed
import os
import pandas as pd #for importing files,  https://pandas.pydata.org/pandas-docs/version/0.18.1/generated/pandas.DataFrame.html
import numpy as np  #for calculations, array manipulations, and fun. In computing, fun is an abreviation for function
import matplotlib.pyplot as plt #for scientifical plots
import matplotlib.cm as cm # for the semisom function
import random #for randomizing indeces
import pylab #correct x and y limits after plotting
import statsmodels.api as sm #for statistic models
import glob #for importing several files to one python matrix
import os #for importing several files to one python matrix
import scipy.stats
import seaborn as sns
from random import randrange, uniform #uniform function makes random float values between a range
from scipy import stats #for statistics
from scipy import * #everything from scipy
from scipy.stats import ks_2samp # comparison of two samples
from scipy.stats import kde
from statsmodels.api import qqplot_2samples #for QQ plots
from numpy import * #e.g isnan command
from itertools import cycle, chain # for efficient iterations
from math import atan2,degrees # for degrees, and angles
from minisom import MiniSom #HERE IS THE SOM FUNCTION
from numpy import genfromtxt,array,linalg,zeros,apply_along_axis 
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

#%% 2) IMPORT MODEL DATA
#COMBES! i.e. EXPLANT CULTURE
dfc_experim2 = pd.read_excel('C:/python/' + 'mmc4.xlsx')
df2_experimi = pd.read_excel('C:/python/' + 'left-speed.xls', header=None)
#https://stackoverflow.com/questions/38884466/how-to-select-a-range-of-values-in-a-pandas-dataframe-column
#%% OULU EXPERIMENTS! i.e. KIDNEY CULTURE
# Oulu 2018 positions->distances, always check the header
df_oulu_x = pd.read_csv('C:/python/position_1_5b3d.csv', delimiter=',', header=None)
df_oulu_y = pd.read_csv('C:/python/position_2_5b3d.csv', delimiter=',', header=None)
df_oulu_z = pd.read_csv('C:/python/position_3_5b3d.csv', delimiter=',', header=None)
df_speed_ilya5b = pd.read_csv('C:/python/speed_5b3d.csv', delimiter=',', header=None)
# Uncomment, if you use these results
#df_oulu_xa = pd.read_csv('C:/python/position_1_7b3d.csv', delimiter=',', header=None)
#df_oulu_ya = pd.read_csv('C:/python/position_2_7b3d.csv', delimiter=',', header=None)
#df_oulu_za = pd.read_csv('C:/python/position_3_7b3d.csv', delimiter=',', header=None)
#df_speed_ilya5b = pd.read_csv('C:/python/speed_7b3d.csv', delimiter=',', header=None)
#MODEL 7 DATA:
dfc_mod_dist = pd.read_csv('distances_pso3d.csv',header='infer')
dfc_mod_vlc  = pd.read_csv('NPcells_avg_velocity_pso3d.csv',header='infer')

#%% 3) CONVERT DATA SUTIABLE FOR SOM, i.e. MAKING OF DSOM MATRIX
#First, DSOM to COMBES (Explant Culture) data:
#d_som=dfc_experim2[['speed','tipendDist','time','x','y','z','dx','dy','dz']]
d_som=dfc_experim2[['speed.xy','time','x','y','z','dx','dy','dz','tipendDist','time_interval_sec', 'sample']]  
d_som=d_som.fillna(0)
d_som['speed']=(np.sqrt(d_som['dx']**2+d_som['dy']**2)/d_som['time_interval_sec'])
d_som['velocity']=(d_som['dx']/d_som['time_interval_sec'])+(d_som['dy']/d_som['time_interval_sec'])
d_tip_x=175
d_tip_y=300
d_corner_x=250
d_corner_y=200
d_som['tip'] =np.sqrt((d_tip_x-d_som['x'])**2+(d_tip_y-d_som['y'])**2)
d_som['corner'] =np.sqrt((d_corner_x-d_som['x'])**2+(d_corner_y-d_som['y'])**2)
d_som=d_som[d_som['tip']<=np.max(d_som['tipendDist'])]
d_som=d_som[d_som['corner']<=np.max(d_som['tipendDist'])]
d_som['tip per corner']=d_som['tip']/d_som['corner'] 

#Second, DSOM to Oulu (Kidney Organoid) data:
df_oulu_x=df_oulu_x.T*0.24
df_oulu_y=df_oulu_y.T*0.24
df_oulu_z=df_oulu_z.T*0.24
df_speed_ilya5b=df_speed_ilya5b.T*0.24 # multiple with the lens factor
df_oulu_x=df_oulu_x.fillna(0)
df_oulu_y=df_oulu_y.fillna(0)
df_oulu_z=df_oulu_z.fillna(0)
df_speed_ilya5b=df_speed_ilya5b.fillna(0)
#https://www.geeksforgeeks.org/python-pandas-dataframe-insert/ (not like this)
dox=[]
doy=[]
doz=[]
dsd=[]
for i in range(0,16):
    dox.append(df_oulu_x[i][df_oulu_x[i] != 0])
    doy.append(df_oulu_y[i][df_oulu_y[i] != 0])
    doz.append(df_oulu_z[i][df_oulu_z[i] != 0])
    dsd.append(df_speed_ilya5b[i][df_speed_ilya5b[i] != 0])
#% Popping the zero elements
dox.pop(15)
doy.pop(15)
doz.pop(15)
dsd.pop(15)
meanx=[]
meany=[]
meanz=[]
means=[]
for i in range(15):
    meanx.append(dox[i].shape)
    meany.append(doy[i].shape)
    meanz.append(doz[i].shape)
    means.append(dsd[i].shape)    
x_cut=round(np.mean(meanx),0)
y_cut=round(np.mean(meany),0)
z_cut=round(np.mean(meanz),0)
s_cut=round(np.mean(means),0)
a=[]
a=[x_cut,y_cut,z_cut,s_cut]
lim=int(min(a)) #4578, this is the cut value
b = np.zeros([len(dox),len(max(dox,key = lambda x: len(x)))])
for i,j in enumerate(dox):
    b[i][0:len(j)] = j
dox=b.T
dox=dox[0:lim,:]
b1 = np.zeros([len(doy),len(max(doy,key = lambda x: len(x)))])
for i,j in enumerate(doy):
    b1[i][0:len(j)] = j
doy=b1.T
doy=doy[0:lim,:]
b2 = np.zeros([len(doz),len(max(doz,key = lambda x: len(x)))])
for i,j in enumerate(doz):
    b2[i][0:len(j)] = j
doz=b2.T
doz=doz[0:lim,:]
b3 = np.zeros([len(dsd),len(max(dsd,key = lambda x: len(x)))])
for i,j in enumerate(dsd):
    b3[i][0:len(j)] = j
dsd=b3.T
dsd=dsd[0:lim,:]
xo=dox.flatten()
yo=doy.flatten()
zo=doz.flatten()
so=dsd.flatten()
d_som=[]
d_som=pd.DataFrame(xo)
d_som["y"]=pd.DataFrame(yo)
d_som["z"]=pd.DataFrame(zo)
d_som["speed"]=pd.DataFrame(so)
#https://stackoverflow.com/questions/7185495/python-range-with-duplicates
timex=np.arange(len(d_som)*lim) // lim
d_som["time"]=pd.DataFrame(timex)
d_som.columns = ["x","y","z","speed","time"]
columnsTitles=["speed","x","y","z","time"]
d_som=d_som.reindex(columns=columnsTitles)
d_min = np.min(d_som[d_som > 0])
# Assign the median to the zero elements 
d_som['x'][d_som['x'] == 0] = d_min['x'] #x was not assigned when you did this..
d_som['y'][d_som['y'] == 0] = d_min['y']
d_som['z'][d_som['z'] == 0] = d_min['z']
d_som['speed'][d_som['speed'] == 0] = d_min['speed']
#d_tip_x=np.mean(d_som[d_som['time']==1]['x']) #0.0034113086634443056
#d_tip_y=np.mean(d_som[d_som['time']==1]['y']) #0.018757923815989494
#d_tip_z=np.mean(d_som[d_som['time']==1]['z']) #0.018757923815989494
#d_corner_x=np.mean(d_som[d_som['time']==np.max(d_som['time'])]['x']) #-0.0035579985360893636
#d_corner_y=np.mean(d_som[d_som['time']==np.max(d_som['time'])]['y']) #0.005485683404759942
#d_corner_z=np.mean(d_som[d_som['time']==np.max(d_som['time'])]['z']) #0.005485683404759942
d_tip_x=0.5
d_tip_y=0.8
d_tip_z=0.5
d_corner_x=0.4
d_corner_y=0.4
d_corner_z=0.5
dvel=np.zeros((len(d_som),1))
for i in range(len(d_som)-1 ):
    dvel[i] = d_som['x'][i+1]-d_som['x'][i]+d_som['y'][i+1]-d_som['y'][i]+d_som['z'][i+1]-d_som['z'][i]
dvel = np.append(0.01, dvel)  
d_som['velocity']=pd.DataFrame(dvel)
d_som['tip'] =np.sqrt((d_tip_x-d_som['x'])**2+
     (d_tip_y-d_som['y'])**2+(d_tip_z-d_som['z'])**2)
d_som['corner'] =np.sqrt((d_corner_x-d_som['x'])**2+
     (d_corner_y-d_som['y'])**2+(d_corner_z-d_som['z'])**2)
d_som['tip']=d_som['tip'].fillna(0)
d_som['corner']=d_som['corner'].fillna(0)
d_som['x']=d_som['x'].fillna(0)
d_som['y']=d_som['y'].fillna(0)
d_som['z']=d_som['z'].fillna(0)
d_som['time']=d_som['time'].fillna(0)
d_som['speed']=d_som['speed'].fillna(0)
d_som['velocity']=d_som['speed'].fillna(0)
d_som.to_csv('C:/python/SOM_REF_EXPS AND MODS/irist.csv', index=False,header=False)
#$.19 to microm per minute in this resolution
#plt.hist(d_som['speed'], 100)*4.17
#plt.hist(d_som['y'], 100) #around 0.2, -0.4 vastaa around 
#x (-0.722544:0.721368)#y (-0.730224:0.69228)
d_som['speed']=d_som['speed']*(1/0.24)
d_som['velocity']=d_som['speed']*(1/0.24)
d_som['x']=((d_som['x']-min(d_som['x']))/(max(d_som['x'])-min(d_som['x'])))
d_som['y']=((d_som['y']-min(d_som['y']))/(max(d_som['y'])-min(d_som['y'])))
d_som['z']=((d_som['z']-min(d_som['z']))/(max(d_som['z'])-min(d_som['z'])))
#d_som['speed']=((d_som['speed']-min(d_som['speed']))/(max(d_som['speed'])-min(d_som['speed'])))
#d_som['velocity']=((d_som['velocity']-min(d_som['velocity']))/(max(d_som['velocity'])-min(d_som['velocity'])))
d_som['tip']=((d_som['tip']-min(d_som['tip']))/(max(d_som['tip'])-min(d_som['tip'])))
d_som['corner']=((d_som['corner']-min(d_som['corner']))/(max(d_som['corner'])-min(d_som['corner'])))
d_som['time']=((d_som['time']-min(d_som['time']))/(max(d_som['time'])-min(d_som['time'])))

#THIRD, DSOM TO MODEL 7 DATA:
#%% MODEL DSOM:
dfc_mod= [dfc_mod_dist,dfc_mod_vlc]
dfc_mod = pd.concat(dfc_mod, axis=1) #works, but time frimes are not the same!!
#https://stackoverflow.com/questions/38884466/how-to-select-a-range-of-values-in-a-pandas-dataframe-column
df_ok2=dfc_mod_dist.set_index("time_mcs")  # this for correct indeces
df_ok=dfc_mod_vlc.set_index("time_mcs")  # this for correct indeces
indeces=np.unique(df_ok2.index)
index = 0
dvel=np.zeros((len(dfc_mod_dist),1))
for i in range(len(dfc_mod_dist)-1):
    dvel[i] = dfc_mod_dist['x_position_(px)'][i+1]-dfc_mod_dist['x_position_(px)'][i]+dfc_mod_dist['y_position_(px)'][i+1]-dfc_mod_dist['y_position_(px)'][i]+dfc_mod_dist['z_position_(px)'][i+1]-dfc_mod_dist['z_position_(px)'][i]
dvel = np.append(0.01, dvel)  
xm=[]
ym=[]
zm=[]
#Speed function:
def inst_fun2(xm,ym,zm,points=101):
    index_points=points           # max cell values
    r=np.zeros((index_points,1)) # here should be the size of the matrix len(...)
    
    for i in range(len(xm)):
            r[i] = np.sqrt(xm[i]**2 + ym[i]**2 + zm[i]**2)

    diffa = np.zeros((index_points-1,1)) # the size of difference is one less...
    diffa=abs(np.diff(r,axis=0)) #y #yes! :) ; this calculates abs(r(t + dt) - r(t))   
    diffa = np.append(0.01, diffa)
        
    return diffa  
xm=dfc_mod_dist['x_position_(px)']
ym=dfc_mod_dist['y_position_(px)']   
zm=dfc_mod_dist['z_position_(px)']
points=len(dfc_mod_dist['x_position_(px)'])   
diffa=inst_fun2(xm,ym,zm,points)    
time=dfc_mod_dist['time_mcs']  
tip=dfc_mod_dist['cell_dist_to_tip'] 
corner=dfc_mod_dist['cell_dist_to_corner'] 
d_som=[]
d_som=pd.DataFrame(diffa)
d_som['speed']=pd.DataFrame(diffa)
d_som['velocity']=pd.DataFrame(dvel)
d_som['x']=pd.DataFrame(xm)
d_som['y']=pd.DataFrame(ym)
d_som['z']=pd.DataFrame(zm)
d_som['time']=pd.DataFrame(time)
d_som['tip']=pd.DataFrame(tip)
d_som['corner']=pd.DataFrame(corner)
d_som['tip']=d_som['tip'].fillna(0)
d_som['corner']=d_som['corner'].fillna(0)
d_som['x']=d_som['x'].fillna(0)
d_som['y']=d_som['y'].fillna(0)
d_som['z']=d_som['z'].fillna(0)
d_som['time']=d_som['time'].fillna(0)
d_som['speed']=d_som['speed'].fillna(0)
d_som['velocity']=d_som['velocity'].fillna(0)
d_som=d_som.ix[:,1:]
d_som.to_csv('C:/python/SOM_REF_EXPS AND MODS/irist.csv',index=False,header=False)
#https://matplotlib.org/gallery/images_contours_and_fields/contourf_demo.html#sphx-glr-gallery-images-contours-and-fields-contourf-demo-py
#https://www.geeksforgeeks.org/python-pandas-dataframe-insert/ (not like this)
#https://stackoverflow.com/questions/7185495/python-range-with-duplicates

#%% 4) CHECK THE CONVERSION OF OULU AND MODEL DATA, AND THE REGIONS OF SPEEDS MANUALLY:
#%% Oulu for new things, 23.5.19:
d_som2=pd.DataFrame()
d_som2['x']=((d_som['x']-min(d_som['x']))/(max(d_som['x'])-min(d_som['x'])))
d_som2['y']=((d_som['y']-min(d_som['y']))/(max(d_som['y'])-min(d_som['y'])))
d_som2['z']=((d_som['z']-min(d_som['z']))/(max(d_som['z'])-min(d_som['z'])))
d_som2['tip']=((d_som['tip']-min(d_som['tip']))/(max(d_som['tip'])-min(d_som['tip'])))
d_som2['corner']=((d_som['corner']-min(d_som['corner']))/(max(d_som['corner'])-min(d_som['corner'])))
d_som2['time']=((d_som['time']-min(d_som['time']))/(max(d_som['time'])-min(d_som['time'])))
b=d_som['tip']/d_som['corner']
plt.plot(b[a])
tipp_d=np.var(d_som['tip'])/np.mean(d_som['tip'])
cornerr_d=np.var(d_som['corner'])/np.mean(d_som['corner'])
s=np.var(d_som['speed'])/np.mean(d_som['speed'])
#%%corner:
aa=0.18<d_som['x']
bb=0.82>d_som['x']
cc=0.36>d_som['y']
cd=0.36<d_som['y']
plt.plot(d_som['time'][aa & bb & cd],d_som['corner'][aa & bb & cd]) 
np.var(d_som['corner'][aa & bb & cd])/np.mean(d_som['corner'][aa & bb & cd]) #0.3428903415233606
np.var(d_som['tip'][cc])/np.mean(d_som['tip'][cc]) #0.5918579669867154

#Analysing Oulu data (before SOM):
#%%Finding the mean speeds of group 2 and 8
#G2: x[0.38;0.48], y[0.51;61] & G8: x[0.58;0.68], y[0.18;0.28]
#https://stackoverflow.com/questions/21415661/logical-operators-for-boolean-indexing-in-pandas
d_somg2x=d_som['x'].loc[(d_som['x'] >=0.38) & (d_som['x'] <=0.48)].index
d_somg2y=d_som['y'].loc[(d_som['y'] >=0.51) & (d_som['y'] <=0.61)].index
dsom_G2_xy=np.intersect1d(np.unique(d_somg2x),np.unique(d_somg2y))
dsom_G2_vel=d_som['velocity'].loc[dsom_G2_xy] #mean is.. 0.18029588786823558 :) np.mean(dsom_G2_vel)
#%%G8
d_somg8x=d_som['x'].loc[(d_som['x'] >=0.58) & (d_som['x'] <=0.68)].index
d_somg8y=d_som['y'].loc[(d_som['y'] >=0.18) & (d_som['y'] <=0.28)].index
dsom_G8_xy=np.intersect1d(np.unique(d_somg8x),np.unique(d_somg8y))
dsom_G8_vel=d_som['velocity'].loc[dsom_G8_xy] #mean is.. 0.24525446176138 :)
plt.hist(d_som['velocity'],500)
plt.xlabel('Velocity / μm/min')
plt.ylabel('Number of Cell Coordinates / NO')
d_vel1=d_som['velocity'].loc[(d_som['velocity'] >=0) & (d_som['velocity'] <=0.2)]
d_vel2=d_som['velocity'].loc[(d_som['velocity'] >=0.2) & (d_som['velocity'] <=0.8)]
np.mean(d_vel1) #len(d_vel1) is 47842
np.mean(d_vel2) #len(d_vel2) is 19894
d_x1=d_som['x'].loc[(d_som['velocity'] >=0.05) & (d_som['velocity'] <=0.15)]
d_y1=d_som['y'].loc[(d_som['velocity'] >=0.05) & (d_som['velocity'] <=0.15)]
np.mean(d_x1) #0.4982
np.mean(d_y1) #0.5228 ,len(d_y1) or len(d_x1) is 34477
d_x2=d_som['x'].loc[(d_som['velocity'] >=0.4) & (d_som['velocity'] <=0.5)]
d_y2=d_som['y'].loc[(d_som['velocity'] >=0.4) & (d_som['velocity'] <=0.5)]
np.mean(d_x2) #0.5000
np.mean(d_y2) #0.5166 #len(d_y2) or len(d_x2) is 5520
# so percentages are for the overall:
per_G2=len(dsom_G2_vel)/len(d_y1) #0.08051744641355106
per_G8=len(dsom_G8_vel)/len(d_y2) #0.1423913043478261
d_som['speed']=d_som['speed']*(60)
d_som['velocity']=d_som['velocity']*(60)
d_som['x']=((d_som['x']-min(d_som['x']))/(max(d_som['x'])-min(d_som['x'])))
d_som['y']=((d_som['y']-min(d_som['y']))/(max(d_som['y'])-min(d_som['y'])))
d_som['z']=((d_som['z']-min(d_som['z']))/(max(d_som['z'])-min(d_som['z'])))
#d_som['speed']=((d_som['speed']-min(d_som['speed']))/(max(d_som['speed'])-min(d_som['speed'])))
#d_som['velocity']=((d_som['velocity']-min(d_som['velocity']))/(max(d_som['velocity'])-min(d_som['velocity'])))
d_som['tip']=((d_som['tip']-min(d_som['tip']))/(max(d_som['tip'])-min(d_som['tip'])))
d_som['corner']=((d_som['corner']-min(d_som['corner']))/(max(d_som['corner'])-min(d_som['corner'])))
d_som['time']=((d_som['time']-min(d_som['time']))/(max(d_som['time'])-min(d_som['time'])))
#%%Finding the mean speeds of group 7 and 5
#G7: x[0.38;0.48], y[0.51;61] & G8: x[0.58;0.68], y[0.18;0.28]
#https://stackoverflow.com/questions/21415661/logical-operators-for-boolean-indexing-in-pandas
d_somg7x=d_som['x'].loc[(d_som['x'] >=0.72) & (d_som['x'] <=0.75)].index #corner
d_somg7y=d_som['y'].loc[(d_som['y'] >=0.4) & (d_som['y'] <=0.43)].index
#d_somg7x=d_som['x'].loc[(d_som['x'] >=0.80) & (d_som['x'] <=0.95)].index #tip
#d_somg7y=d_som['y'].loc[(d_som['y'] >=0.80) & (d_som['y'] <=0.85)].index
#np.unique(d_somg2x)
#%% G7
dsom_G7_xy=np.intersect1d(np.unique(d_somg7x),np.unique(d_somg7y))
dsom_G7_vel=d_som['speed'].loc[dsom_G7_xy] #mean is.. 0.19070308159970456 :) np.mean(dsom_G7_vel) , len is 20..
#%%G5
d_somg5x=d_som['x'].loc[(d_som['x'] >=0.75) & (d_som['x'] <=0.95)].index #corner or tip
d_somg5y=d_som['y'].loc[(d_som['y'] >=0.05) & (d_som['y'] <=0.2)].index
dsom_G5_xy=np.intersect1d(np.unique(d_somg5x),np.unique(d_somg5y))
dsom_G5_vel=d_som['speed'].loc[dsom_G5_xy] #mean is.. 0.1364 np.mean(dsom_G5_vel) #, 253
plt.hist(d_som['speed'],100)
plt.xlabel('Speed / μm/min')
plt.ylabel('Number of Cell Coordinates / NO')
#%%len(dsom_G5_vel)
d_vel1=d_som['speed'].loc[(d_som['speed'] >=0) & (d_som['speed'] <=0.15)]
d_vel2=d_som['speed'].loc[(d_som['speed'] >=0.15) & (d_som['speed'] <=0.4)]
np.mean(d_vel1) #0.08065244450408424, len(d_vel1) is 2397
np.mean(d_vel2) #0.23371330236663115, len(d_vel2) is 1263
d_x1=d_som['x'].loc[(d_som['speed'] >=0.05) & (d_som['speed'] <=0.15)]
d_y1=d_som['y'].loc[(d_som['speed'] >=0.05) & (d_som['speed'] <=0.15)]
np.mean(d_x1) #0.4982
np.mean(d_y1) #0.5228 ,len(d_y1) or len(d_x1) is 34477
d_x2=d_som['x'].loc[(d_som['speed'] >=0.15) & (d_som['speed'] <=0.3)]
d_y2=d_som['y'].loc[(d_som['speed'] >=0.15) & (d_som['speed'] <=0.3)]
# so percentages are for the overall:
per_G7=len(dsom_G7_vel)/len(d_y1)
per_G5=len(dsom_G5_vel)/len(d_y2)

#%% Procedures to model speeds
d_som['speed']=d_som['speed']*(1/60)
d_som['velocity']=d_som['speed']*(1/60)
d_som['x']=((d_som['x']-min(d_som['x']))/(max(d_som['x'])-min(d_som['x'])))
d_som['y']=((d_som['y']-min(d_som['y']))/(max(d_som['y'])-min(d_som['y'])))
d_som['z']=((d_som['z']-min(d_som['z']))/(max(d_som['z'])-min(d_som['z'])))
#d_som['speed']=((d_som['speed']-min(d_som['speed']))/(max(d_som['speed'])-min(d_som['speed'])))
#d_som['velocity']=((d_som['velocity']-min(d_som['velocity']))/(max(d_som['velocity'])-min(d_som['velocity'])))
d_som['tip']=((d_som['tip']-min(d_som['tip']))/(max(d_som['tip'])-min(d_som['tip'])))
d_som['corner']=((d_som['corner']-min(d_som['corner']))/(max(d_som['corner'])-min(d_som['corner'])))
d_som['time']=((d_som['time']-min(d_som['time']))/(max(d_som['time'])-min(d_som['time'])))
#%%Finding the mean speeds of group 2 and 8
#G2: x[0.25;0.35], y[0.35;0.45] & G8: x[0.7;0.8], y[0.65;0.75]
#https://stackoverflow.com/questions/21415661/logical-operators-for-boolean-indexing-in-pandas
d_somg2x=d_som['x'].loc[(d_som['x'] >=0.1) & (d_som['x'] <=0.4)].index
d_somg2y=d_som['y'].loc[(d_som['y'] >=0.3) & (d_som['y'] <=0.5)].index
dsom_G2_xy=np.intersect1d(np.unique(d_somg2x),np.unique(d_somg2y))
dsom_G2_vel=d_som['speed'].loc[dsom_G2_xy] #mean is.. 0.22384762253706825 :) np.mean(dsom_G2_vel)

#%%G8
d_somg8x=d_som['x'].loc[(d_som['x'] >=0.5) & (d_som['x'] <=0.9)].index
d_somg8y=d_som['y'].loc[(d_som['y'] >=0.5) & (d_som['y'] <=0.7)].index
dsom_G8_xy=np.intersect1d(np.unique(d_somg8x),np.unique(d_somg8y))
dsom_G8_vel=d_som['speed'].loc[dsom_G8_xy] #mean is.. 0.14481119948402452 :) np.mean(dsom_G8_vel)
plt.hist(d_som['speed'],250)
plt.xlabel('Velocity / μm/min')
plt.ylabel('Number of Cell Coordinates / NO')
d_vel1=d_som['speed'].loc[(d_som['speed'] >=0) & (d_som['speed'] <=0.18)]
d_vel2=d_som['speed'].loc[(d_som['speed'] >=0.18) & (d_som['speed'] <=0.5)]
np.mean(d_vel1) #0.07726795170354336, len(d_vel1) is 25980
np.mean(d_vel2) #0.28903318503862235, len(d_vel2) is 11857
d_x1=d_som['x'].loc[(d_som['speed'] >=0.1) & (d_som['speed'] <=0.2)]
d_y1=d_som['y'].loc[(d_som['speed'] >=0.1) & (d_som['speed'] <=0.2)]
np.mean(d_x1) #0.4982
np.mean(d_y1) #0.5228 ,len(d_y1) or len(d_x1) is 34477
d_x2=d_som['x'].loc[(d_som['speed'] >=0.17) & (d_som['speed'] <=0.27)]
d_y2=d_som['y'].loc[(d_som['speed'] >=0.17) & (d_som['speed'] <=0.27)]
# so percentages are for the overall:
per_G8=len(dsom_G8_vel)/len(d_y1)
per_G2=len(dsom_G2_vel)/len(d_y2)

#%%Check always separately:
plt.hist(d_som['tipendDist'].fillna(0), bins=20)
#2d velocity:https://physics.stackexchange.com/questions/98363/the-velocity-formula-mathbfv-mathbfu-mathbfat-for-1d-2d-3d-wh
#https://matplotlib.org/gallery/images_contours_and_fields/contourf_demo.html#sphx-glr-gallery-images-contours-and-fields-contourf-demo-py
#% This works, 241118
#https://github.com/JustGlowing/minisom
#https://dzone.com/articles/self-organizing-maps
#https://scikit-learn.org/stable/

#%% 5) PERFORM THE MINISOM FOR DSOMs' csvs, e.g. irst.csv, AND PLOT THE NODE PLOT
#%% Here is the SOM for the experimental and my data
data = np.genfromtxt('C:/python/SOM_REF_EXPS AND MODS/irist.csv', delimiter=',', usecols=(1,2,3,4))
#check the used columns (nos) per dataset, model or various experiments
#% data normalization
data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)
# Initialization and training
som = MiniSom(10, 10, 4, sigma=4, learning_rate=0.5, 
              neighborhood_function='mexican_hat', random_seed=10)
#for mexican hat:seed=10, sigma=2, learning_rate=0.3 are good, and are also used elshewhere
#size from 7 to 16 for all
som.random_weights_init(data)
som.pca_weights_init(data)
print("Training...")
som.train_random(data, 5000)  # random training
print("\n...ready!")
#%If you have already trained, your SOM is ready, and you may want to print the figure instead...
plt.figure(figsize=(10, 10))
# Plotting the response for each pattern in the iris dataset
plt.pcolor(som.distance_map().T, cmap="RdYlGn")  # plotting the distance map as background, rainbow is good
plt.colorbar()
#https://stackoverflow.com/questions/1841565/valueerror-invalid-literal-for-int-with-base-10
#t=list(range(1,(len(d_som)+1)))
t=list(range(1,(len(d_som)+1)))
#check the size of markers and colors...
markers =(['s', 'D','o', ',', '.','s','D']*5000+['*']*4592)
#https://stackoverflow.com/questions/13091649/unique-plot-marker-for-each-plot-in-matplotlib
colors =(['C1']*5000+['C2']*5000+['C3']*5000+
         ['C4']*5000+['C5']*5000+['C6']*5000+['C7']*5000+['C8']*4592)
for cnt, xx in enumerate(data):
    w = som.winner(xx)  # getting the winner
    # palce a marker on the winning position for the sample xx
    plt.plot(w[0]+.5, w[1]+.5, markers[t[cnt]-1], markerfacecolor='None',
             markeredgecolor=colors[t[cnt]-1], markersize=12, markeredgewidth=3)
plt.axis([0,som._weights.shape[0],0,som._weights.shape[1]]) #this works!
#https://github.com/JustGlowing/minisom/issues/17
plt.savefig('C:/python/SOM_REF_EXPS AND MODS/SOM_REFERENCE_velocity_and_location_4.1.2019_mod.png') #do not use pyplot
#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
plt.show() # show the figure
#https://stackoverflow.com/questions/10017876/matplotlib-markers-disappear-when-edgecolor-none
#check the length of markers and edgecolors len(markers[t[cnt]-1]), and insert -1 if necessary

#%% 6) GROUP THE SOM DATA:
#%% Grouping function for the resulting data:
def groups(data,d_som,amount=8):
    #%
    a=[]
    b=[]
    for i in range(len(data)):
        a.append(som.winner(data[i]))
#    https://docs.python.org/3.3/tutorial/datastructures.html
    #%This is how yuo get all groups...
    aa=pd.DataFrame(a)
    #tuple(range(0,9))
    aa.columns = ['a', 'b']
    aa.index=d_som.index
    oon=[]
    for i in range(0,9):
        for j in range(0,9):
            oon.append(aa.loc[(aa['a'] == i) & (aa['b'] == j)])   
    ll=[]
    for i in range(len(oon)):
        ll.append(np.shape(oon[i]))
        lll=pd.DataFrame(ll)
    lll.columns = ['a', 'b']
    d=(-np.sort(-lll['a'],axis=0))[0:amount] #descending sort
    dd=[]
    d2=[]
    for i in range(len(oon)):
        if int(np.shape(oon[i])[0]) >= int(min(d)):
            dd.append(int(np.unique(oon[i]['a'])))
            d2.append(int(np.unique(oon[i]['b'])))
    dd=pd.DataFrame(dd)
    d2=pd.DataFrame(d2)
    dd['d2'] = d2
    dd.columns = ['dd', 'd2']
    for i in range(amount):
        ooon.append(aa.loc[(aa['a'] == dd.ix[i,0]) & (aa['b'] == dd.ix[i,1])])
#    oo1=aa.loc[(aa['a'] == dd.ix[0,0]) & (aa['b'] == dd.ix[0,1])] #ok, so this is how you do it, not with 'trans'
#    oo2=aa.loc[(aa['a'] == dd.ix[1,0]) & (aa['b'] == dd.ix[1,1])] #ok,
#    oo3=aa.loc[(aa['a'] == dd.ix[1,0]) & (aa['b'] == dd.ix[2,1])] #ok, 
#    oo4=aa.loc[(aa['a'] == dd.ix[3,0]) & (aa['b'] == dd.ix[3,1])] #ok, 
#    oo5=aa.loc[(aa['a'] == dd.ix[4,0]) & (aa['b'] == dd.ix[4,1])] #ok,    
#%    ooon=[oo1,oo2,oo3,oo4,oo5]    
    return ooon,d

#%Select half times for better group selections:
def showx_half(d_som,ooon):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    #Variables defined
    ftime=[]
    mid=[]
    mid2=[]
    ft1=[]
    ft2=[]
    ooz1=[]
    ooz2=[]
    ap=[]
    ftn=[]
    ft2o=np.zeros((len(ooon),1))
    ftime2=[]
    
    #Needed auxialiry funtion for finding right middle point index:
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    for i in range(len(ooon)):
        ftime.append(np.unique(d_som.ix[list(ooon[i].index),'time'])) #t
        ftime2.append(ftime[i][~np.isnan(ftime[i])])
        ap.append(d_som.ix[list(ooon[i].index),:])
        mid.append(int(round(median(ftime2[i]),0)))
        mid2.append(find_nearest(ftime2[i], np.median(ftime2[i]))) #auxialiry is used here
        ft1.append(ftime2[i][0:mid[i]])
        ooz1.append(ap[i].loc[ap[i]['time'].isin(ft1[i])]) #yes!! 
        #these two 'selections' below are not obvious!
        ft2o[i]=pd.DataFrame(ftime2[i]).ix[ftime2[i]==mid2[i],:].index[0] 
        ooz2.append(ap[i].loc[ap[i]['time'].isin(ftime2[i][int(ft2o[i]):])]) #yes!!
    return ooz1,ooz2  

#%% 7) Show original data with SOM groups. Prints multiple different images and histogram    
def showx(d_som,ooon,ooz1,ooz2):
    #Variables exported and scaled to one
    x=[]
    y=[]
    z=[]
    colors=[]
    dist=[]
    T=[] 
    on_index1=[]
    on_index2=[]
    on_index3=[]
    for i in range(len(ooon)):
        on_index1.append(list(ooon[i].index))
        on_index2.append(list(ooz1[i].index))
        on_index3.append(list(ooz2[i].index))
    on_indexi=on_index1+on_index2+on_index3
    mt=np.zeros((len(on_indexi),1))
    mc=np.zeros((len(on_indexi),1))
    md=np.zeros((len(on_indexi),1))
    st=np.zeros((len(on_indexi),1))
    for i in range(len(on_indexi)):
        x.append(np.array(d_som.ix[on_indexi[i],'x']))
        y.append(np.array(d_som.ix[on_indexi[i],'y']))
        z.append(np.array(d_som.ix[on_indexi[i],'z']))
        colors.append(np.array(d_som.ix[on_indexi[i],'speed']))
        dist.append(np.array(d_som.ix[on_indexi[i],'tip']))
        T.append(np.array(d_som.ix[on_indexi[i],'time']))
    for i in range(len(on_indexi)):
        x[i] = x[i]/max(x[i])
        y[i] = y[i]/max(y[i])
        z[i] = z[i]/max(z[i])
        colors[i] = colors[i]/max(colors[i])
        dist[i] = dist[i]/max(dist[i])
        T[i]=T[i]/max(T[i])
    nx=[]
    ny=[]
    nz=[]
    nc=[]
    nd=[]
    nt=[]
#    https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
    for i in range(len(on_indexi)): 
        nx.append((x[i]-min(x[i]))/(max(x[i])-min(x[i])))
        ny.append((y[i]-min(y[i]))/(max(y[i])-min(y[i])))
        nz.append((z[i]-min(z[i]))/(max(z[i])-min(z[i])))
        nc.append((colors[i]-min(colors[i]))/(max(colors[i])-min(colors[i])))
        nd.append((dist[i]-min(dist[i]))/(max(dist[i])-min(dist[i]))) 
        nt.append((T[i]-min(T[i]))/(max(T[i])-min(T[i]))) 
    x=nx
    y=ny
    z=nz
    colors=nc
    dist=nd
    T=nt
        #%Rounding for plotting purposes
    for i in range(len(on_indexi)):
        for j in range(len(on_indexi[i])):
            if x[i][j] >= 0.99:
                x[i][j]=0.99
            if x[i][j] <= 0.01:
                x[i][j]=0.01
            if y[i][j] >= 0.99:
                y[i][j]=0.99
            if y[i][j] <= 0.01:
                y[i][j]=0.01    
            if z[i][j] >= 0.99:
                z[i][j]=0.99
            if z[i][j] <= 0.01:
                z[i][j]=0.01 
            if colors[i][j] >= 0.99:
                colors[i][j]=0.99
            if colors[i][j] <= 0.01:
                colors[i][j]=0.01
            if T[i][j] >= np.max(T[i]):
                T[i][j]=np.max(T[i])
            if T[i][j] <= np.min(T[i]):
                T[i][j]=np.min(T[i])
            if dist[i][j] >= 0.99:
                dist[i][j]=0.99
            if dist[i][j] <= 0.01:
                dist[i][j]=0.01
        x[i]=np.round(x[i],2)
        y[i]=np.round(y[i],2)
        z[i]=np.round(z[i],2)    
        colors[i]=np.round(colors[i],2) 
        dist[i]=np.round(dist[i],2) 
        T[i]=np.round(T[i],1)
    st_n=[]
    for i in range(len(x)):
        st_n.append(len(x[i]))  
    weight_s=max(st_n)      
    for i in range(len(on_indexi)):
        mt[i]=np.mean(T[i])
        mc[i]=np.mean(colors[i])
        md[i]=np.mean(dist[i])
        st[i]=len(x[i])/weight_s*1000 #np.std(T[i])*100 
    mt1=[]
    mc1=[]
    md1=[]
    st1=[]
    for i in range(len(on_indexi)):
        mt1.append(float(mt[i]))
        mc1.append(float(mc[i]))
        md1.append(float(md[i]))
        st1.append(float(st[i]))
#%https://stackoverflow.com/questions/20196159/how-to-append-multiple-values-to-a-list-in-python
#https://www.thegeekstuff.com/2013/06/python-list/?utm_source=feedly
    fig1 = plt.figure()
    ax = plt.axes(projection='3d')
    label=['Group 1','Group 2','Group 3','Group 4','Group 5','Group 6','Group 7','Group 8',
   'Group 1 Early',' Group 2 Early','Group 3 Early',' Group 4 Early','Group 5 Early','Group 6 Early',
   'Group 7 Early','Group 8 Early',
   'Group 1 Late','Group 2 Late',' Group 3 Late','Group 4 Late','Group 5 Late','Group 6 Late',
   'Group 7 Late','Group 8 Late',]
    #    colors = cm.rainbow(np.linspace(0, 1, len(mt)))
    col=['blue','green','yellow','red','black','brown','silver','magenta',
         'blue','green','yellow','red','black','brown','silver','magenta',
         'blue','green','yellow','red','black','brown','silver','magenta']
    alpha=[0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,
           0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,
           0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35]
    for i in range(len(mt)):
        ax.scatter3D(mt1[i], mc1[i], md1[i],s=st1[i],c=col[i], alpha = alpha[i],label=label[i])
    ax.set_xlabel('Time / AU')
    ax.set_ylabel('Speed normed/ AU')
    ax.set_zlabel('Distance normed/ AU')
    ax.set_title('SOM cluster groups')
    legend = plt.legend(loc='upper left', shadow=False, fontsize='xx-small',markerfirst=True,markerscale=0.2)
    #https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('w')
    pylab.savefig('SOM_Groups.png')
    plt.show()
    fig2 = plt.figure()
    label=[
       'G1 Early',' G2 Early','G3 Early',' G4 Early','G5 Early','G6 Early',
       'G7 Early','G8 Early',
       'G1 Late','G2 Late',' G3 Late','G4 Late','G5 Late','G6 Late',
       'G7 Late','G8 Late',]  
    col=[
         'blue','green','yellow','red','black','brown','silver','magenta',
         'blue','green','yellow','red','black','brown','silver','magenta']
    alpha=[
           0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,
           0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]
    for i in range(16):
        plt.scatter(mt[8:24][i], mc[8:24][i],s=st[8:24][i],c=col[i], alpha = alpha[i],label=label[i])
    plt.xlabel('Time / AU')
    plt.ylabel('Speed normed/ AU')
    plt.title('SOM cluster groups')
#    legend = plt.legend(loc='lower center', shadow=True, fontsize='x-small',markerfirst=True,markerscale=0.3)
#    legend.get_frame().set_facecolor('w')
    pylab.savefig('SOM_Time splitted to two groups_time and speed plotv2.png')
    plt.show()
    fig3 = plt.figure()
    for i in range(16):
        plt.scatter(mt[8:24][i], md[8:24][i],s=st[8:24][i],c=col[i], alpha = alpha[i],label=label[i])
    plt.xlabel('Time / AU')
    plt.ylabel('Distance / AU')
    plt.title('SOM cluster groups')
    pylab.savefig('SOM_Time splitted to two groups_time and distance plotv2.png')
    plt.show()
    fig4 = plt.figure()
    for i in range(16):
        plt.scatter(md[8:24][i], mc[8:24][i],s=st[8:24][i],c=col[i], alpha = alpha[i],label=label[i])
    plt.xlabel('Distance / AU')
    plt.ylabel('Speed / AU')
    pylab.savefig('SOM_Time splitted to two groups_distance and speed plot.png')
    plt.show()
    td=[]
    x1=x[8:24]
    y1=y[8:24]
    z1=z[8:24]
    colors1=colors[8:24]
    T1=T[8:24]
    dist1=dist[8:24]   
    for i in range(len(x1)):
        td.append(pd.DataFrame({
                "X normed / AU": x1[i],
                "Y normed / AU": y1[i],
                "Z normed / AU": z1[i],
                "Speed normed / AU": colors1[i],
                "Time normed / AU": T1[i]
                }))   
    fig5, axn = plt.subplots(4, 4, figsize=(10, 10),sharex=True, sharey=True)
    sns.despine(left=True)
    cbar_ax = fig5.add_axes([.91, .3, .03, .4])
    for i, ax in enumerate(axn.flat):
        sns.jointplot(x=td[i-1]["X normed / AU"], #note the math!! was not obvious
                y=td[i-1]["Y normed / AU"], 
                ax=ax,
                cbar=i == 0,
                vmin=0, vmax=5,
                kind='kde',color="skyblue",
                cbar_ax=None if i else cbar_ax,
                cbar_kws={"ticks":[0,2,5]})  
    fig5.suptitle("Distances and speeds from SOM groups by velocity, and time of Combes et al. NP cells", 
                 fontsize=11, fontweight=0, color='black', style='italic', y=1)
    # Axis title
    fig5.text(0.5, 0.01, 'Distance normed / AU', ha='center', va='baseline')
    fig5.text(0.01, 0.5, 'Speed normed / AU', ha='center', va='baseline', rotation='vertical')
    fig5.savefig('Distance and speed_tot.png')
    pylab.savefig('Distance and speed.png')
    fig6, axn = plt.subplots(4, 4, figsize=(10, 10),sharex=True, sharey=True)
    sns.despine(left=True)
    cbar_ax = fig6.add_axes([.91, .3, .03, .4]) 
    for i, ax in enumerate(axn.flat):
        sns.jointplot(x=td[i-1]["Time normed / AU"], #note the math!! was not obvious
            y=td[i-1]["Speed normed / AU"], 
            ax=ax,
            cbar=i == 0,
            vmin=0, vmax=3,
            kind='kde',color="skyblue",
            cbar_ax=None if i else cbar_ax,
            cbar_kws={"ticks":[0,1,3]})         
    fig6.suptitle("Times and speeds from SOM groups by velocity, and time of Combes et al. NP cells", 
                 fontsize=11, fontweight=0, color='black', style='italic', y=1)
    # Axis title
    fig6.text(0.5, 0.01, 'Time normed / AU', ha='center', va='baseline')
    fig6.text(0.01, 0.5, 'Speed normed / AU', ha='center', va='baseline', rotation='vertical')
    fig6.savefig('Speed and time_tot.png')
    pylab.savefig('Speed and time.png')
    fig7, axn = plt.subplots(4, 4, figsize=(10, 10),sharex=True, sharey=True)
    sns.despine(left=True)
    cbar_ax = fig7.add_axes([.91, .3, .03, .4])
    for i, ax in enumerate(axn.flat):
        sns.jointplot(x=td[i-1]["X normed / AU"], #note the math!! was not obvious
            y=td[i-1]["Speed normed / AU"], 
            ax=ax,
            cbar=i == 0,
            vmin=0, vmax=5,
            kind='kde',color="skyblue",
            cbar_ax=None if i else cbar_ax,
            cbar_kws={"ticks":[0,2,5]})  
    fig7.suptitle("Xs and speeds from SOM groups by velocity, distacne, and time of Combes et al. NP cells", 
                 fontsize=11, fontweight=0, color='black', style='italic', y=1)
    # Axis title
    fig7.text(0.5, 0.01, 'X normed / AU', ha='center', va='baseline')
    fig7.text(0.01, 0.5, 'Speed normed / AU', ha='center', va='baseline', rotation='vertical')
    fig7.savefig('X and speed_tot.png')
    fig8, axn = plt.subplots(4, 4, figsize=(10, 10),sharex=True, sharey=True)
    sns.despine(left=True)
    cbar_ax = fig8.add_axes([.91, .3, .03, .4])
    for i, ax in enumerate(axn.flat):
        sns.jointplot(x=td[i-1]["X normed / AU"], #note the math!! was not obvious
            y=td[i-1]["Y normed / AU"], 
            ax=ax,
            cbar=i == 0,
            vmin=0, vmax=5,
            kind='kde',color="skyblue",
            cbar_ax=None if i else cbar_ax,
            cbar_kws={"ticks":[0,2.5,5]})
        
    fig8.suptitle("XY locations from SOM groups by velocity, and time of Combes et al. NP cells", 
                 fontsize=11, fontweight=0, color='black', style='italic', y=1)
    # Axis title
    fig8.text(0.5, 0.01, 'X normed / AU', ha='center', va='baseline')
    fig8.text(0.01, 0.5, 'Y normed / AU', ha='center', va='baseline', rotation='vertical')
    fig8.savefig('XY locations_tot.png')
    fig9, axn = plt.subplots(4, 4, figsize=(10, 10),sharex=True, sharey=True)
    sns.despine(left=True)
    cbar_ax = fig9.add_axes([.91, .3, .03, .4])
    for i, ax in enumerate(axn.flat):
        sns.jointplot(x=td[i-1]["X normed / AU"], #note the math!! was not obvious
            y=td[i-1]["Z normed / AU"], 
            ax=ax,
            cbar=i == 0,
            vmin=0, vmax=5,
            kind='kde',color="skyblue",
            cbar_ax=None if i else cbar_ax,
            cbar_kws={"ticks":[0,2.5,5]})
    
    fig9.suptitle("XZ locations from SOM groups by velocity, and time of Combes et al. NP cells", 
                 fontsize=13, fontweight=0, color='black', style='italic', y=1)
    # Axis title
    fig9.text(0.5, 0.01, 'X normed / AU', ha='center', va='baseline')
    fig9.text(0.01, 0.5, 'Z normed / AU', ha='center', va='baseline', rotation='vertical')
    fig9.savefig('XZ locations_tot.png')
    pylab.savefig('XZ locations.png')
    return(x,y,z,T,dist,colors,on_indexi,mt,md,mc,st)
    
def show(d_som,on_indexi):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    #%Variables exported and scaled to one
    x=np.array(d_som.ix[on_indexi,'x'])
    y=np.array(d_som.ix[on_indexi,'y'])
    z=np.array(d_som.ix[on_indexi,'z'])
    colors=np.array(d_som.ix[on_indexi,'speed'])
    dist=np.array(d_som.ix[on_indexi,'tip'])
    T=np.array(d_som.ix[on_indexi,'time'])
    x = x/max(x)
    y = y/max(y)
    z = z/max(z)
    colors = colors/max(colors)
    dist = dist/max(dist)
    T=T/max(T)
        #%
    nx=[]
    ny=[]
    nz=[]
    nc=[]
    nd=[]
    nt=[]
    nx=((x-min(x))/(max(x)-min(x)))
    ny=((y-min(y))/(max(y)-min(y)))
    nz=((z-min(z))/(max(z)-min(z)))
    nc=((colors-min(colors))/(max(colors)-min(colors)))
    nd=((dist-min(dist))/(max(dist)-min(dist))) 
    nt=((T-min(T))/(max(T)-min(T))) 
    x=nx
    y=ny
    z=nz
    colors=nc
    dist=nd
    T=nt
    #Rounding for plotting purposes
    for i in range(len(y)):
        if x[i] >= 0.99:
            x[i]=0.99
        if x[i] <= 0.01:
            x[i]=0.01
        if y[i] >= 0.99:
            y[i]=0.99
        if y[i] <= 0.01:
            y[i]=0.01    
        if z[i] >= 0.99:
            z[i]=0.99
        if z[i] <= 0.01:
            z[i]=0.01 
        if colors[i] >= 0.99:
            colors[i]=0.99
        if colors[i] <= 0.01:
            colors[i]=0.01
        if T[i] >= 0.99:
            T[i]=0.99
        if T[i] <= 0.01:
            T[i]=0.01
        if dist[i] >= 0.99:
            dist[i]=0.99
        if dist[i] <= 0.01:
            dist[i]=0.01
    x=np.round(x,2)
    y=np.round(y,2)
    z=np.round(z,2)    
    colors=np.round(colors,2) 
    dist=np.round(dist,2) 
    T=np.round(T,2) 
    #%$Histogramming unique times and speeds from Combes for first group    
    ftime=np.unique(d_som.ix[on_indexi,'time'])
    ftime2=ftime[~np.isnan(ftime)]
    #%time: len(x): 69 fx is not the same time as in d_som
    ooz=d_som.loc[d_som['time'].isin(ftime2)] #yes!!
    mean_y=[]
    for ind in ftime2:
        mean_y.append(np.mean(ooz['speed'].loc[d_som['time']==ind])) 
    mean_y=np.array(mean_y)
    mean_y=mean_y[~np.isnan(mean_y)]
    meany=np.mean(mean_y)
    sty=np.std(mean_y)
    #%
    fig1=plt.figure()
    plt.xlabel('Speeds')
    plt.ylabel('Frequency') 
    bins=int(round(len(mean_y)/10)+3*abs(round(log(np.mean(mean_y))))) 
    plt.hist(mean_y, bins=bins)
    pylab.savefig('hist_speed.png')
    fig1.show()
    #%
    fig2=plt.figure()
    plt.xlabel('Experimental time / 15min/unit')
    plt.ylabel('Speeds / micrometers/min') 
    plt.plot(mean_y[1:])
    pylab.savefig('plot_speed.png')
    fig2.show()
    test_df2 = pd.DataFrame({
            "X normed / AU": x,
            "Speed normed / AU": colors,
            "Speed potential normed / AU": colors
        })
    fig, ax = plt.subplots()
#    plt.axis([(min(x)-round(var(x)*4,2)), (max(x)+round(var(x)*4,2)), 
#              (min(colors)-round(var(colors)*4,2)), (max(colors)+round(var(colors)*4,2))])   
    result2 = test_df2.sort_values(by=["Speed potential normed / AU"],ascending=True)
    #for i in range(len(test_df)):
    result2.plot(kind="scatter", x="X normed / AU", y="Speed normed / AU", s=10,
                 c="Speed normed / AU", cmap='RdYlGn', ax=ax)     
    pylab.savefig('plot_speed and x.png')
    fig.show()    
    np.savetxt('file_numpy.txt', (x,y,z,dist,T,colors))
    np.savetxt('file_numpy2.txt', (meany,sty))
#    http://docs.hyperion-rt.org/en/stable/tutorials/python_writing.html
    #load with:heihou=np.loadtxt('file_numpy.txt')
#    https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html#numpy.loadtxt
    fig4=plt.figure()
    plt.hist2d(x, y, bins=15, cmap='RdYlGn',label='Cell amount')
    plt.colorbar().set_label('Cell amount', rotation=90)
#    https://python-graph-gallery.com/83-basic-2d-histograms-with-matplotlib/
#    https://python-graph-gallery.com/2d-density-plot/
    plt.xlabel('X / AU')
    plt.ylabel('Y/ AU')
    plt.title('Cell amount')
#    https://matplotlib.org/gallery/images_contours_and_fields/contourf_demo.html#sphx-glr-gallery-images-contours-and-fields-contourf-demo-py
    pylab.savefig('2d histogram of cells in x, y.png')
    fig4.show()
    fig5=plt.figure()
    plt.hist2d(y, z, bins=15, cmap='RdYlGn',label='Cell amount')
    plt.colorbar().set_label('Cell amount', rotation=90)
    plt.xlabel('Y / AU')
    plt.ylabel('Z/ AU')
    plt.title('Cell amount')
    pylab.savefig('2d histogram of cells in y, z.png')
    fig5.show()
    fig6=plt.figure()
    plt.xlabel('Cell X, or Y  Coordinate')
    plt.ylabel('Cells at the Coordinate') 
    bins=int(round(len(mean_y)/10)+3*abs(round(log(np.mean(mean_y))))) 
    plt.hist(y, bins=15,edgecolor='None',ls='dotted', lw=3, alpha = 0.9,color='black',label='Y')
    plt.hist(z, bins=15,edgecolor='None', ls='dashed', lw=3,alpha = 0.7,color='blue',label='Z')
    plt.hist(x, bins=15,edgecolor='None', lw=3,alpha = 0.8,color='green',label='X')
    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C7')
    pylab.savefig('histogram of cells of x, y, and z.png')
    fig6.show()
    fig7 = plt.figure()
    ax = plt.axes(projection='3d')
    p=ax.scatter3D(T,colors, x, 
             s=20,c=colors, cmap='RdYlGn')
    ax.get_yaxis().labelpad = 1   
    ax.get_xaxis().labelpad = 1
    from matplotlib import rcParams
    rcParams['axes.labelpad'] = 1
    #legend
    cbar = fig7.colorbar(p)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.set_ylabel('Speed normed / AU', rotation=90)
    ax.set_xlabel('Time / AU')
    ax.set_ylabel('Speed / AU')
    ax.set_zlabel('X / AU')
    ax.set_title('SOM cluster group / AU')
    pylab.savefig('SOM group.s real time, speed, and x.png')
    fig7.show()
    test_df = pd.DataFrame({
        "X normed / AU": x,
        "Y normed / AU": y,
        "Z normed / AU": z,
        "Time normed / AU": T,            
        "Speed normed / AU": colors
    })
    result = test_df.sort_values(by=["Speed normed / AU"],ascending=True)
    fig8 = plt.figure()
    ax = plt.axes(projection='3d')
    p=ax.scatter3D(result["X normed / AU"], result["Y normed / AU"], result["Z normed / AU"], 
                 s=20,c=result["Speed normed / AU"], cmap='RdYlGn')
    ax.get_yaxis().labelpad = 1   
    ax.get_xaxis().labelpad = 1
    from matplotlib import rcParams
    rcParams['axes.labelpad'] = 1
    #legend
    cbar = fig8.colorbar(p)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.set_ylabel('Speed normed / AU', rotation=90)
    ax.set_xlabel('X normed / AU')
    ax.set_ylabel('Y normed / AU')
    ax.set_zlabel('Z normed / AU')
    ax.set_title('SOM cluster group')
    pylab.savefig('SOM group''s speeds in real space.png')
    fig8.show()    
    return(x,y,T,colors,meany,sty)    
#https://stackoverflow.com/questions/43121584/matplotlib-scatterplot-x-axis-labels
#https://seaborn.pydata.org/tutorial/color_palettes.html
#%%https://matplotlib.org/users/colormaps.html    
x= mmfx
y= mmfy
z= mmfz
colors= mmfs  
    
#%% 8) DO THE SEMISOM PLOT, i.e. average cell coordinates with average speeds, normed [0,1]
def semi_som(x,y,z,colors,condition=True, c2="RdBu"):
    ##https://matplotlib.org/users/pyplot_tutorial.html
    norm = [(float(i)-min(colors))/(max(colors)-min(colors)) for i in colors]
    xn = [(float(i)-min(x))/(max(x)-min(x)) for i in x]
    yn = [(float(i)-min(y))/(max(y)-min(y)) for i in y]
    zn = [(float(i)-min(z))/(max(z)-min(z)) for i in z]
    for i in range(len(zn)):
        if zn[i] <= 0.05:
            zn[i]=0.05
        if zn[i] >= 0.95:
            zn[i]=0.95   
    for i in range(len(zn)):
        zn[i]=100*(zn[i])
    ##https://stackoverflow.com/questions/7994394/efficient-thresholding-filter-of-an-array-with-numpy
    test_df = pd.DataFrame({
            "X normed / AU": xn,
            "Y normed / AU": yn,
            
            "Speed potential normed / AU": norm
        })
    fig, ax = plt.subplots()
    result = test_df.sort_values(by=["Speed potential normed / AU"],ascending=condition)   
    result.plot(kind="scatter", x="X normed / AU", y="Y normed / AU", s=zn,
                 c="Speed potential normed / AU", cmap=c2, ax=ax, alpha=0.6)
    pylab.savefig('Cell coordinates and speeds in real space_v2.png')  
#    https://pandas.pydata.org/pandas-docs/version/0.23.0/visualization.html#scatter-plot
#    https://python-graph-gallery.com/83-basic-2d-histograms-with-matplotlib/
#    https://stackoverflow.com/questions/40766909/suggestions-to-plot-overlapping-lines-in-matplotlib
    return result
#%%Test 16419, seismic is the blue to red (via white.. coolwarm would be via mix, gist_ncar, rainbow)
#https://matplotlib.org/examples/color/colormaps_reference.html    
semi_som(x,y,z,colors,condition=False, c2="seismic")

#%% 9) Executing the analysis with previous functions:
ooon=[]
ooz1=[]
ooz2=[]
ooon,d=groups(data,d_som,amount=8) 
#replace df_testt2 with your original data vector including speeds and locations of cell etc. unit
ooz1,ooz2=showx_half(d_som,ooon) 
#replace df_testt2 with your vector..
#%% This does most of the basic work:
x,y,z,T,dist,colors,on_indexi,mt,md,mc,st=showx(d_som,ooon,ooz1,ooz2) #similarly as above
#colors differences:
mtime=mc[8:24]
mt1=mtime[0:8]
mt2=mtime[8:16]
#%% Here are the group NOs that you should check, 
#i.e. ones that are time independent, i.e. end time is close enough to start time:
#%this will tell about the spatial speed locations
a=[]
def speed_loc(mt1,mt2):
    for i in range(len(mt1)):
        if (mt1[i]*1.01 >= mt2[i]) or (mt2[i] <= mt1[i]*0.99): 
            #the value ranges should be higher for model (model not as good as experiments?), 
            #better 0.9 to 1.1, than 0.99 to 1.01
        #+-1% seems to limit the groups adequately, together with check z,y,z distributions, 
#        and it may limit 1-2 groups still, until you have just 'two locational groups'
#    if False or True:
            a.append(i+1)
            print(i+1)
            
    return a
a=speed_loc(mt1,mt2)    

#%%Analysing the groups one-by-one, semisom and esp. the speed coordinate plots of groups
#https://stackoverflow.com/questions/431684/how-do-i-change-directory-cd-in-python
path=['C:/python/SOM_REF_EXPS AND MODS/Group1','C:/python/SOM_REF_EXPS AND MODS/Group2','C:/python/SOM_REF_EXPS AND MODS/Group3','C:/python/SOM_REF_EXPS AND MODS/Group4','C:/python/SOM_REF_EXPS AND MODS/Group5',
      'C:/python/SOM_REF_EXPS AND MODS/Group6','C:/python/SOM_REF_EXPS AND MODS/Group7','C:/python/SOM_REF_EXPS AND MODS/Group8',
      'C:/python/SOM_REF_EXPS AND MODS/Group1e','C:/python/SOM_REF_EXPS AND MODS/Group2e','C:/python/SOM_REF_EXPS AND MODS/Group3e','C:/python/SOM_REF_EXPS AND MODS/Group4e','C:/python/SOM_REF_EXPS AND MODS/Group5e',
      'C:/python/SOM_REF_EXPS AND MODS/Group6e','C:/python/SOM_REF_EXPS AND MODS/Group7e','C:/python/SOM_REF_EXPS AND MODS/Group8e',
      'C:/python/SOM_REF_EXPS AND MODS/Group1l','C:/python/SOM_REF_EXPS AND MODS/Group2l','C:/python/SOM_REF_EXPS AND MODS/Group3l','C:/python/SOM_REF_EXPS AND MODS/Group4l','C:/python/SOM_REF_EXPS AND MODS/Group5l',
      'C:/python/SOM_REF_EXPS AND MODS/Group6l','C:/python/SOM_REF_EXPS AND MODS/Group7l','C:/python/SOM_REF_EXPS AND MODS/Group8l']
#https://stackoverflow.com/questions/13819496/what-is-different-between-makedirs-and-mkdir-of-os
td=[]
for i in range(len(x)):
    os.mkdir(path[i])
    os.chdir(path[i])
    show(d_som,on_indexi[i])
    semi_som(x[i],y[i],z[i],colors[i],condition=True, c2="RdYlGn")   
    td=[]
    td.append(pd.DataFrame({
            "X normed / AU": x[i],
            "Y normed / AU": y[i],
            "Z normed / AU": z[i],
            "Speed normed / AU": colors[i],
            "Time normed / AU": T[i],
            "Distance normed / AU": dist[i]
            }))   
    for i in range(len(td)):    
        sns.set(style="white", color_codes=True)
        sns.jointplot(x=td[i]["X normed / AU"], y=td[i]["Y normed / AU"], kind='kde', color="skyblue")
        pylab.savefig("Contour_xy.png")
        sns.jointplot(x=td[i]["X normed / AU"], y=td[i]["Z normed / AU"], kind='kde', color="skyblue")
        pylab.savefig("Contour_xz.png")
        sns.jointplot(x=td[i]["X normed / AU"], y=td[i]["Speed normed / AU"], kind='kde', color="skyblue")
        pylab.savefig("Contour_x_speed.png")
        sns.jointplot(x=td[i]["Time normed / AU"], y=td[i]["Speed normed / AU"], kind='kde', color="skyblue")
        pylab.savefig("Contour_time_speed.png")
        sns.jointplot(x=td[i]["Distance normed / AU"], y=td[i]["Speed normed / AU"], kind='kde', color="skyblue")
        pylab.savefig("Contour_distance_speed.png")

# 10) SAVE THE DATA AND/OR FOR (RE)ANALYSING
#%%works, save a as string:
with open("variable_file.txt", "w") as variable_file:
    variable_file.write(str(a)) # a was the 'end time is close enough to start time' variable matrix
#https://stackoverflow.com/questions/30139243/saving-a-variable-in-a-text-file
#%%Save total groups (also late and early)
import pickle
with open('ooon', 'wb') as fp:
    pickle.dump(ooon, fp)    
with open('ooz1', 'wb') as fp:
    pickle.dump(ooz1, fp)    
with open('ooz2', 'wb') as fp:
    pickle.dump(ooz2, fp)
with open('d_som', 'wb') as fp:
    pickle.dump(d_som, fp)    
#%%If needed to reanalyse, os.chdir("C:/python/analysis2"), open 
os.chdir("C:/python/" )   #check the directory  
def pickle(file='ooz1'):     
    import pickle
    with open(file, 'rb') as fp:
        itemlist1 = pickle.load(fp) #Model Name (e.g. reference), corner values for NP and MM  
    return(itemlist1)   
#https://www.programcreek.com/python/example/190/pickle.load   
#These are the most important variables:
ooon=pickle(file='ooon') # 
ooz1=pickle(file='ooz1') #  
ooz2=pickle(file='ooz2') # 
d_som=pickle(file='d_som')          
#I may need to save some of these too, otherwise I may not do extra plotting...   
#https://stackoverflow.com/questions/28356359/one-colorbar-for-seaborn-heatmaps-in-subplot
#https://stackoverflow.com/questions/44076339/seaborn-clustermap-set-colorbar-ticks
