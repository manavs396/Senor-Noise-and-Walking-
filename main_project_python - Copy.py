#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import math
import matplotlib
from scipy.fftpack import fft
from math import pow
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import seaborn as sb
import scipy
from scipy import signal
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from statsmodels.nonparametric.smoothers_lowess import lowess 
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# # Helper Function 

# In[2]:


# Butter worth filter 

def filter_butterworth(temp):
    
    butter, atemp = signal.butter(3, 0.05, btype='lowpass', analog=False)
    filter_a      = signal.filtfilt(butter, atemp, temp)
    
    return filter_a 

def butterworth(x):
    
    butter, atemp = signal.butter(8, 0.05, btype='lowpass', analog=False)
    filter_b      = signal.filtfilt(butter, atemp, x)
    
    return filter_b

def calculate_butterworth(read_variable):
    
    filtere = read_variable.apply(Butterworth_filter(read_variable))
    stat    = calc_summary(filtere)
    
    return stat

def acc_butterworth():
    
    read_read_variableSignal = acc_read_variable["ax"]
    b, a                     = signal.butter(3, 0.03, btype='lowpass', analog=False)
    filter_a                 = signal.filtfilt(b, a, read_read_variableSignal)
    
    return filter_a


#calculate absolute acceleration 

def absolute_function(x,y,z):
    
    total = np.sqrt(x*x + y*y + z*z)
   
    return total 

def acc_bessellow():
    
    raw_signal = acc_read_variable["ax"]
    d, c       = signal.bessel(3, 0.03, 'low', analog=False, norm='phase')
    result     = signal.filtfilt(d, c, raw_signal)
    
    return result

#lowess filter
def loweesfilter():
    
    low_smooth = lowess(acc_read_variable["ax"], np.arange(acc_read_variable["time"].shape[0]), frac=0.01)
    
    return low_smooth

def kalmanfileter(): 
    
    # Read read_variable
    var_a = acc_read_variable[["ax"]].values
    kf    = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    kf    = kf.em(var_a, n_iter=5)
    (filtered_state_means, filtered_state_covariances) = kf.filter(acc_read_variable["ax"])
    
    return kf

def height_count():
   
    # calculate max_height
    counter = 0 
    temp_max_height, _   = signal.find_peaks(variable1, prominence  = 0.5)
    temp_shape, _        = signal.find_peaks(-variable1, prominence = 0.5)
    
    temp_figure          = matplotlib.pyplot.gcf()
    
    plt.plot(variable1, label ="Leveling the read data ", color = 'Orange')
    
    plt.plot(temp_max_height, variable1[temp_max_height], "+", label="Max-height")
    
    #plt.plot(temp_shape, variable1[temp_shape], "-", label="Narrow Shape")
    
    plt.plot(np.zeros_like(variable1), "--", color="#A2142F")

    #counter++
    #print("Height Maximum got ={}".format(len(temp_max_height)+len(temp_shape)))

#https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
def calculate_distance(lat,lon): 
   
    R    = 6371; # Radius of the earth in km
    dLat = ((lat.shift(-1) - lat).dropna())*(np.pi/180);
    dLon = ((lon.shift(-1) - lon).dropna())*(np.pi/180); 
    a    = np.sin(dLat/2) * np.sin(dLat/2) + np.cos((lat)*(np.pi/180)) * np.cos((lat.shift(-1)))*(np.pi/180) * np.sin(dLon/2) * np.sin(dLon/2)
    
    c    = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)); 
    d    = R * c * 1000; # Distance in meters)
    
    return d

# https://stackoverflow.com/questions/26245539/i-need-to-write-a-program-in-python-to-calculate-the-calories-burned-while-doing    
def calories_count(acceleration_avg ):
    
    
    temp_force       = 90* acceleration_avg 
    
    # Converting into meters 
    temp_distance    = (0.38)/0.00062137
    temp_work        = temp_force*temp_distance
    
    # Converting into calories 
    result_calories  = (temp_work)/4184 
    
    return result_calories

#FFT analysis frequency of steps
def fourier_transform_x_dataset(timeInitial,timeEnd, count_samples):
    
    # get number of samples
    var_x         = np.linspace(0.0,count_samples * ((timeEnd - timeInitial)/count_samples), count_samples)
    
    fft_y         = scipy.fft(np.sin(50.0 * 2.0*np.pi*var_x) + 0.5*np.sin(80.0 * 2.0*np.pi*var_x))

    fft_x         = np.linspace(0.0, 1.0/(2.0*((timeEnd -timeInitial)/count_samples)), count_samples//2)
    
    return fft_x

def fourier_transform_y_dataset(timeInitial,timeEnd, count_samples):
    
    # get number of samples
    var_x         = np.linspace(0.0,count_samples * ((timeEnd - timeInitial)/count_samples), count_samples)
    
    fft_y         = scipy.fft(np.sin(50.0 * 2.0*np.pi*var_x) + 0.5*np.sin(80.0 * 2.0*np.pi*var_x))
   
    fft_x         = np.linspace(0.0, 1.0/(2.0*((timeEnd -timeInitial)/count_samples)), count_samples//2)
    
    return fft_y


# # Reading, Cleaning data from different csv set for Comparison

# In[3]:


# Reading the variable from CSV files 
read_variable        = pd.read_csv("lhs_swing.csv")
read_variable1       = pd.read_csv("knee_left.csv")

read_variable2       = pd.read_csv("drive_left.csv")
read_variable3       = pd.read_csv("drive_right.csv")

read_variable4       = pd.read_csv("stairs_left.csv")
read_variable5       = pd.read_csv("knee_left.csv")

read_variable6       = pd.read_csv("left_hand.csv")
read_variable7       = pd.read_csv("drive_left.csv")

# For acceleration 
acc_read_variable = pd.read_csv("Gps_left.csv")

read_variable['ax']  = filter_butterworth(read_variable['ax'])
read_variable1['ax']  = filter_butterworth(read_variable1['ax'])
read_variable2['ax']  = filter_butterworth(read_variable2['ax'])
read_variable3['ax']  = filter_butterworth(read_variable3['ax'])
read_variable4['ax']  = filter_butterworth(read_variable4['ax'])
read_variable5['ax']  = filter_butterworth(read_variable5['ax'])
read_variable6['ax']  = filter_butterworth(read_variable6['ax'])
read_variable7['ax']  = filter_butterworth(read_variable7['ax'])

read_variable['ay']  = filter_butterworth(read_variable['ay'])
read_variable1['ay']  = filter_butterworth(read_variable1['ay'])
read_variable2['ay']  = filter_butterworth(read_variable2['ay'])
read_variable3['ay']  = filter_butterworth(read_variable3['ay'])
read_variable4['ay']  = filter_butterworth(read_variable4['ay'])
read_variable5['ay']  = filter_butterworth(read_variable5['ay'])
read_variable6['ay'] = filter_butterworth(read_variable6['ay'])
read_variable7['ay'] = filter_butterworth(read_variable7['ay'])


read_variable['az'] = filter_butterworth(read_variable['az'])
read_variable1['az'] = filter_butterworth(read_variable1['az'])
read_variable2['az'] = filter_butterworth(read_variable2['az'])
read_variable3['az'] = filter_butterworth(read_variable3['az'])
read_variable4['az'] = filter_butterworth(read_variable4['az'])
read_variable5['az'] = filter_butterworth(read_variable5['az'])
read_variable6['az'] = filter_butterworth(read_variable6['az'])
read_variable7['az'] = filter_butterworth(read_variable7['az'])

acc_read_variable['ax'] = filter_butterworth(acc_read_variable['ax'])
acc_read_variable['ay'] = filter_butterworth(acc_read_variable['ay'])
acc_read_variable['az'] = filter_butterworth(acc_read_variable['az'])
acc_read_variable = acc_read_variable.loc[:, ~acc_read_variable.columns.str.contains('^Unnamed')]
acc_read_variable['gFx'] = filter_butterworth(acc_read_variable['gFx'])
acc_read_variable['gFy'] = filter_butterworth(acc_read_variable['gFy'])
acc_read_variable['gFz'] = filter_butterworth(acc_read_variable['gFz'])
acc_read_variable = acc_read_variable.drop(acc_read_variable[acc_read_variable.Latitude == 0.0].index)


# In[4]:


# read_variable For acceleration containing latitude and longitude (GPS)
acc_read_variable


# In[5]:


# Depend upon the data set 
read_variable = read_variable.drop(columns=['gFx', 'gFy','gFz','wx','wy','wz','P'])
read_variable1 = read_variable1.drop(columns=['gFx', 'gFy','gFz','wx','wy','wz','P'])
#read_variable = read_variable.drop(columns=['acc','P'])
#read_variable1 = read_variable1.drop(columns=['acc','P'])


# In[6]:


read_variable


# In[7]:


read_variable1


# # Acceleration Vs Time

# In[8]:


# Graph of accerlration x-axis and time 
read_variable['ax'] = filter_butterworth(read_variable['ax'])

plt.scatter(read_variable['time'],read_variable['ay'],5, color = '#00FF00')

plt.title('Accelerations in X-axis')
plt.xlabel('time')
plt.ylabel('acceleration')
plt.suptitle("ButterWorth Filter")
plt.suptitle('Figure-0 (ButterWorth Filter)')
plt.show


# In[9]:


# Graph for accerlration y-axis and time 
read_variable['ay'] = filter_butterworth(read_variable['ay'])

plt.scatter(read_variable['time'],read_variable['ay'],5, color = '#0000FF')

plt.title('Accelerations in Y-axis')
plt.xlabel('time')
plt.ylabel('acceleration')
plt.suptitle("ButterWorth Filter")
plt.suptitle('Figure-1 (ButterWorth Filter)')
plt.show


# In[10]:


# Graph for accerlration z-axis and time 
read_variable['az'] = filter_butterworth(read_variable['az'])

plt.scatter(read_variable['time'],read_variable['az'],5,color = '#D95319')
plt.title('Accelerations in Z-axis')
plt.xlabel('time')
plt.ylabel('acceleration')
plt.suptitle("ButterWorth Filter")
plt.suptitle('Figure-2 (ButterWorth Filter)')

plt.show


# In[11]:


# Tri-acceleration of ax,ay,az with respect to time 
read_variable = read_variable[read_variable['time']>2]
read_variable = read_variable[read_variable['time']<20]

plt.plot(read_variable['time'],read_variable['ax'],color='#00FF00')

plt.plot(read_variable['time'],read_variable['ay'],color='#0000FF')

plt.plot(read_variable['time'],read_variable['az'],color='#D95319')

plt.title('x-axis accelerations(Green), y-axis acceleration(Blue), z-axis acceleration(orange)')
plt.xlabel('time')
plt.ylabel('Acceleration')
plt.suptitle('Figure-3')

plt.show


# # Tri-Acceleration Vs Time (ButterWorth Filter)

# In[12]:


# Tri-acceleration of ax,ay,az with respect to time Using Butter filter 
read_variable['ax'] = filter_butterworth(read_variable['ax'])
read_variable['ay'] = filter_butterworth(read_variable['ay'])
read_variable['az'] = filter_butterworth(read_variable['az'])

plt.scatter(read_variable['time'],read_variable['ax'],5,color='#00FF00')

plt.scatter(read_variable['time'],read_variable['ay'],5, color = '#0000FF')

plt.scatter(read_variable['time'],read_variable['az'],5,color='#D95319')
plt.suptitle("Figure-4 ButterWorth Filter")
plt.title('x-axis accelerations(Green), y-axis acceleration(Blue), z-axis acceleration(orange)')
plt.xlabel('Time')
plt.ylabel('Acceleration')


# In[13]:


read_variable1 = read_variable1[read_variable1['time']>2]
read_variable1 = read_variable1[read_variable1['time']<20]


# # Comparison of means with respect to absolute acceleration for different Data set

# In[14]:


# Total acceleration and comparison  
read_variable['acc']=absolute_function(read_variable['ax'],read_variable['ay'],read_variable['az'])
read_variable1['acc']=absolute_function(read_variable1['ax'],read_variable1['ay'],read_variable1['az'])

read_variable2['acc']=absolute_function(read_variable2['ax'],read_variable2['ay'],read_variable2['az'])
read_variable3['acc']=absolute_function(read_variable3['ax'],read_variable3['ay'],read_variable3['az'])

read_variable4['acc']=absolute_function(read_variable4['ax'],read_variable4['ay'],read_variable4['az'])
read_variable5['acc']=absolute_function(read_variable5['ax'],read_variable5['ay'],read_variable5['az'])

read_variable6['acc']=absolute_function(read_variable6['ax'],read_variable6['ay'],read_variable6['az'])
read_variable7['acc']=absolute_function(read_variable7['ax'],read_variable7['ay'],read_variable7['az'])

x = read_variable['acc'].mean()
y = read_variable1['acc'].mean()

x1 = read_variable2['acc'].mean()
y1 = read_variable3['acc'].mean()

x2 = read_variable4['acc'].mean()
y2 = read_variable5['acc'].mean()

x3 = read_variable6['acc'].mean()
y3 = read_variable7['acc'].mean()

print(x,y)
print(x1,y1)
print(x2,y2)
print(x3,y3)

# As the acceleration of taking phone as a swing in the hand is faster than phone attached to the knee in the first case 


# # Graphical Comparison using acceleration vs time using different dataset

# In[15]:


# Graph of Absolute acceleration vs time data set 1 and data set 2 
plt.scatter(read_variable['time'],read_variable['acc'], 5, color = '#00FF00')

plt.scatter(read_variable1['time'],read_variable1['acc'], 5, color = '#A2142F')
plt.title('Different position Absolute Acceleration: Data Set1: Green, Data Set2: Brown')

plt.xlabel('time')

plt.ylabel('acceleration')
plt.suptitle('Figure-5')


# In[16]:


# Dataset 3 and Dataset 4 
plt.scatter(read_variable2['time'],read_variable2['acc'], 5, color = '#EDB120')

plt.scatter(read_variable3['time'],read_variable3['acc'], 5, color = '#7E2F8E')
plt.title('Different position Absolute Acceleration: Data Set3: Dark Yellow, Data Set4: Purple')

plt.xlabel('time')

plt.ylabel('acceleration')
plt.suptitle('Figure-6')


# In[17]:


# # Dataset 5 and Dataset 6 
plt.scatter(read_variable4['time'],read_variable4['acc'], 5, color = '#FF00FF')

plt.scatter(read_variable5['time'],read_variable5['acc'], 5, color = '#FFFF00')
plt.title('Different position Absolute Acceleration: Data Set5: Pink, Data Set6: Yellow')

plt.xlabel('time')

plt.ylabel('acceleration')
plt.suptitle('Figure-7')


# In[18]:


# Dataset 7 and Dataset 8 
plt.scatter(read_variable6['time'],read_variable6['acc'], 5, color = '#4DBEEE')

plt.scatter(read_variable7['time'],read_variable7['acc'], 5, color = '#D95319')

plt.title('Different position Absolute Acceleration: Data Set7: Blue, Data Set8: Orange')

plt.xlabel('time')

plt.ylabel('acceleration')
plt.suptitle('Figure-8')


# # Absolute Acceleration vs Time (ButterWorth filter)

# In[19]:


# Applying Butterfilter in absolute acceleration 
read_variable['acc']    = butterworth(read_variable['acc'])
read_variable1['acc']   = butterworth(read_variable1['acc'])

plt.plot(read_variable['time'],read_variable['acc'],color='blue')
plt.plot(read_variable1['time'],read_variable1['acc'],color='red')
plt.suptitle('Figure-9')


# # Applying Fourier Transform 

# In[20]:


# Fourier Transform Over Data Set 1 
timeInitial   = read_variable["time"].values[0]
timeEnd       = read_variable["time"].values[-1]
count_samples = read_variable["time"].shape[0]

xf = fourier_transform_x_dataset(timeInitial ,timeEnd, count_samples)
yf =  fourier_transform_y_dataset(timeInitial ,timeEnd, count_samples)
plt.title('Fourier Transform on Data Set1: Left Hand Side swing walking')

plt.scatter(xf, 2.0/read_variable['time'].shape[0] * np.abs(yf[0:read_variable['time'].shape[0]//2]), 5, color = '#A2142F')
plt.suptitle('Figure-10')


# In[21]:


# Fourier Transform Over Data Set 2 
timeInitial   = read_variable1["time"].values[0]
timeEnd       = read_variable1["time"].values[-1]
count_samples = read_variable["time"].shape[0]

xf = fourier_transform_x_dataset(timeInitial, timeEnd, count_samples)
yf = fourier_transform_y_dataset(timeInitial, timeEnd, count_samples)

plt.title('Fourier Transform on dataSet2: collected at knee walking')
plt.ylabel('frequency')

plt.scatter(xf, 2.0/read_variable['time'].shape[0] * np.abs(yf[0:read_variable['time'].shape[0]//2]), 5, color = '#7E2F8E')
plt.suptitle('Figure-11')


# In[22]:


# Fourier Transform Over Data Set 3
timeInitial   = acc_read_variable["time"].values[0]
timeEnd       = acc_read_variable["time"].values[-1]
count_samples = acc_read_variable["time"].shape[0]
xf = fourier_transform_x_dataset(timeInitial, timeEnd, count_samples)
yf = fourier_transform_y_dataset(timeInitial, timeEnd,count_samples)
plt.title('Fourier Transform on dataSet3: collected at left hand walking')
plt.ylabel('frequency')

plt.scatter(xf, 2.0/acc_read_variable['time'].shape[0] * np.abs(yf[0:acc_read_variable['time'].shape[0]//2]), 5, color = '#EDB120')
plt.suptitle('Figure-12')


# # Comparison different Filters (Acceleration vs Time)

# In[23]:


variable1 =  acc_butterworth()


# In[24]:


# Comparing the filters for acceleration vs Time 
kf = kalmanfileter()
(filtered_state_means, filtered_state_covariances) = kf.filter(acc_read_variable["ax"])

low_smooth = loweesfilter()
    
fig = matplotlib.pyplot.gcf()

plt.scatter(acc_read_variable["time"], acc_butterworth(), 5, color = '#D95319')

plt.xlabel("Y-axis, Time")
plt.ylabel("X-axis, Acceleration")

plt.title('Butterworth-Filter')
plt.suptitle('Figure-13')
plt.show()


# In[25]:


# Bessel-filter 
plt.scatter(acc_read_variable["time"], acc_bessellow(), 5, color = '#4DBEEE')
plt.xlabel("Y-axis, Time")
plt.ylabel("X-axis, Acceleration")

plt.title('Bessel-Filter')
plt.suptitle('Figure-14')
plt.show()


# In[26]:


# Lowess-Filter 
plt.scatter(acc_read_variable["time"], low_smooth[:,1],5, color = '#00FF00')
plt.xlabel("Y-axis, Time")
plt.ylabel("X-axis, Acceleration")

plt.title('Lowess-Filter')
plt.suptitle('Figure-15')
plt.show()


# In[27]:


# Kalman Filter 
plt.scatter(acc_read_variable["time"], filtered_state_means, 5, color = '#0000FF')
plt.xlabel("Y-axis, Time")
plt.ylabel("X-axis, Acceleration")

plt.title('Kalman-Filter')
plt.suptitle('Figure-16')
plt.show()


# # Analyse the better filter, used for Max height and Smooth curve 

# In[28]:


# Above we got the efficent filter method which will use to calculate maximum height,and the smooth Curve in the data  
# Here n depend upon the data set  
height_count()

plt.ylabel("Y-axis Angular Velocity")
plt.xlabel("tim")
plt.title("Max-height and Narrow Shape for n step Walking read_variable")
plt.legend()
plt.suptitle('Figure-17')


# # Apply Fourier Transform Max Height in smooth curve

# In[29]:


# Fourier Transform is used for calculating maximum height from the smoothed signal 
timeInitial   = acc_read_variable["time"].values[0]
timeEnd       = acc_read_variable["time"].values[-1]
count_samples = acc_read_variable["time"].shape[0]
xf = fourier_transform_x_dataset(timeInitial, timeEnd, count_samples)
yf = fourier_transform_y_dataset(timeInitial, timeEnd, count_samples)

plt.scatter(xf, 2.0/acc_read_variable['time'].shape[0] * np.abs(yf[0:acc_read_variable['time'].shape[0]//2]), 5, color = '#0000FF')

plt.title('Fourier Transform on dataSet3: collected at Left walking (angular velocity)')
plt.suptitle('Figure-18 (Amplitude vs Frequency)')
plt.show()


# # Differentiate between different data set (1. Calories 2. Acceleration)

# In[30]:


# For Calories Help in differenciate the data set 
# READ THE DATA
running_read_variable=pd.read_csv("lhs_swing.csv")

running_read_variable['ax'] = filter_butterworth(running_read_variable['ax'])
running_read_variable['ay'] = filter_butterworth(running_read_variable['ay'])
running_read_variable['az'] = filter_butterworth(running_read_variable['az'])


# In[31]:


#running_read_variable = running_read_variable.drop(columns=['gFx', 'gFy','gFz','wx','wy','wz'])
running_read_variable['acc']=absolute_function(running_read_variable['ax'],running_read_variable['ay'],running_read_variable['az'])
running_read_variable['az'] = running_read_variable['az'].abs()
read_variable2['az'] = read_variable2['az'].abs()
read_variable3['az'] = read_variable3['az'].abs()
read_variable4['az'] = read_variable4['az'].abs()
read_variable5['az'] = read_variable5['az'].abs()
read_variable6['az'] = read_variable6['az'].abs()
read_variable7['az'] = read_variable7['az'].abs()


# In[32]:


acceleration_avg = running_read_variable['az'].mean()
print("Calorie for left swing:")
print (calories_count(acceleration_avg))


# In[33]:


acceleration_avg = read_variable1['az'].mean()
print("Calorie for walking left knee:")
print (calories_count(acceleration_avg))


# In[34]:


acceleration_avg = read_variable2['az'].mean()
print("Calorie for drive left :")
print (calories_count(acceleration_avg))


# In[35]:


acceleration_avg = read_variable3['az'].mean()
print("Calorie for drive right :")
print (calories_count(acceleration_avg))


# In[36]:


acceleration_avg = read_variable4['az'].mean()
print("Calorie for stairs left :")
print (calories_count(acceleration_avg))


# In[37]:


acceleration_avg = read_variable5['az'].mean()
print("Calorie for knee left :")
print (calories_count(acceleration_avg))


# In[38]:


acceleration_avg = read_variable6['az'].mean()
print("Calorie for left hand:")
print (calories_count(acceleration_avg))


# In[39]:


acceleration_avg = read_variable7['az'].mean()
print("Calorie for drive left:")
print (calories_count(acceleration_avg))


# In[40]:


running_read_variable['Calories'] = calories_count(acceleration_avg)


# # Linear Regression

# In[41]:


# Accuracy check- Machine learning tool for Calories As we used acceleration so linear regression will do the work 
X = running_read_variable[['az']]
y = running_read_variable['Calories']
model = LinearRegression(fit_intercept=True)
model.fit(X, y)

print(model.coef_[0], model.intercept_)


# In[42]:


# Calculating distance from absolute sum 


# In[43]:


acc_read_variable['Absolute_sum']=absolute_function(acc_read_variable['Latitude'],acc_read_variable['Longitude'],0)
calculate_distance(acc_read_variable['Latitude'], acc_read_variable['Longitude']).sum()


# In[44]:


acc_read_variable


# # Walking Speed (Accelerometre)

# In[45]:


# Calculating the walking speed from accelerometre 
# https://electronics.stackexchange.com/questions/112421/measuring-speed-with-3axis-accelerometer
acc_read_variable1 = pd.read_csv("Gps_left.csv")
acc_read_variable2 = pd.read_csv("Gps_right.csv")


acc_read_variable1 = acc_read_variable1.loc[:, ~acc_read_variable1.columns.str.contains('^Unnamed')]
acc_read_variable2 = acc_read_variable2.loc[:, ~acc_read_variable2.columns.str.contains('^Unnamed')]

#acc_read_variable1.ax.abs()
#acc_read_variable1.ay.abs()
#acc_read_variable1.az.abs()

acc_read_variable1['velocity_x'] = 0
acc_read_variable1['velocity_y'] = 0
acc_read_variable1['velocity_z'] = 0
acc_read_variable1['velocity_x'] += acc_read_variable1['ax']*acc_read_variable1['time'] 
acc_read_variable1['velocity_y'] += acc_read_variable1['ay']*acc_read_variable1['time'] 
acc_read_variable1['velocity_z'] += acc_read_variable1['az']*acc_read_variable1['time'] 

acc_read_variable2['velocity_x'] = 0
acc_read_variable2['velocity_y'] = 0
acc_read_variable2['velocity_z'] = 0
acc_read_variable2['velocity_x'] += acc_read_variable2['ax']*acc_read_variable2['time'] 
acc_read_variable2['velocity_y'] += acc_read_variable2['ay']*acc_read_variable2['time'] 
acc_read_variable2['velocity_z'] += acc_read_variable2['az']*acc_read_variable2['time'] 


# In[46]:


#https://physics.stackexchange.com/questions/153159/calculate-speed-from-accelerometer
acc_read_variable1['Speed'] = (acc_read_variable1['velocity_x']*acc_read_variable1['velocity_x'] + acc_read_variable1['velocity_y']*acc_read_variable1['velocity_y']+ acc_read_variable1['velocity_z']*acc_read_variable1['velocity_z']) 
acc_read_variable2['Speed'] = (acc_read_variable2['velocity_x']*acc_read_variable2['velocity_x'] + acc_read_variable2['velocity_y']*acc_read_variable2['velocity_y']+ acc_read_variable2['velocity_z']*acc_read_variable2['velocity_z']) 


# In[47]:


#https://stackoverflow.com/questions/37256540/applying-sqrt-function-on-a-column
acc_read_variable1['Speed'] =acc_read_variable1['Speed'].pow(1./2)
acc_read_variable2['Speed'] =acc_read_variable2['Speed'].pow(1./2)


# In[48]:


acc_read_variable1 


# In[49]:


#Graph comp. of Walking speed and data set1 
plt.plot(acc_read_variable1['time'],acc_read_variable1['Speed'],color='blue')
plt.xlabel('Time')
plt.ylabel("Walking Speed")
plt.title("Left hand vs time")
plt.suptitle('Figure-19')


# In[50]:


#Graph comp. of Walking speed and data set2 
plt.plot(acc_read_variable2['time'],acc_read_variable2['Speed'],color='red')
plt.xlabel('Time')
plt.ylabel("Walking Speed")
plt.title("Right hand vs time")
plt.suptitle('Figure-20')


# # Machine Learning Tools 

# In[51]:


# Accuracy check- Machine learning tool 
X = acc_read_variable1[['time','ax', 'ay','az']]
y = acc_read_variable1['Speed (m/s)']

lab_enc = preprocessing.LabelEncoder()
X.ax = lab_enc.fit_transform(X.ax)
X.ay = lab_enc.fit_transform(X.ay)
X.az = lab_enc.fit_transform(X.az)
X.time = lab_enc.fit_transform(X.time)

y = lab_enc.fit_transform(y)
X_train, X_valid,y_train, y_valid = train_test_split(X,y)

#Random Forest Classifier 

model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10)                  
model.fit(X_train,y_train)
model.score(X_valid, y_valid)
y_predict = model.predict(X_valid)
#test_result = rfc.predict(read_variable_test)
#print(result)


print(accuracy_score(y_valid, y_predict))


# # RandomForestClassifier

# In[52]:


print("For GPS left")
print(model.fit(X_train,y_train))
print(model.score(X_valid, y_valid))


# # Gaussian

# 

# In[53]:


# Gaussian 
model = GaussianNB()
model.fit(X_train, y_train)
y_predict = model.predict(X_valid)
print("For GPS left")
print(model.score(X_train, y_train))
#print(model.score(X_valid, y_valid))

print(accuracy_score(y_valid, y_predict))


# # K-nearest neighbours classifier

# In[54]:


# K-nearest neighbours classifier.
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_predict = model.predict(X_valid)
print("For GPS left walk")
print(model.score(X_train, y_train))
#print(model.score(X_valid, y_valid))
print(accuracy_score(y_valid, y_predict))


# # Standard Scaler , KNeighborsClassifier

# In[55]:


# Standard Scaler , KNeighborsClassifier
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=9)
)
model.fit(X_train, y_train)
y_predict = model.predict(X_valid)
print("For GPS left walk")
print(model.score(X_train, y_train))
#print(model.score(X_valid, y_valid))
print(accuracy_score(y_valid, y_predict))


# # neural network regressor

# In[56]:


# neural network regressor
model = MLPRegressor(hidden_layer_sizes=(8, 6),
                     activation='logistic', solver='lbfgs')
model.fit(X_train, y_train)
print("For GPS left walk")
print(model.score(X_train, y_train))

print(model.score(X_valid, y_valid))


# # RandomForestRegressor

# In[57]:


#  RandomForestRegressor
model = RandomForestRegressor(30, max_depth=4)
model.fit(X_train, y_train)
print("For GPS left walk")
y_predict = model.predict(X_valid)
print(model.score(X_train, y_train))
print(model.score(X_valid, y_valid))
#print(accuracy_score(y_valid, y_predict))


# # K-Neighbor Regressor 

# In[58]:


# K-Neighbor Regressor 
model = KNeighborsRegressor(5)
model.fit(X_train, y_train)
print("For GPS left walk")
y_predict = model.predict(X_valid)
print(model.score(X_train, y_train))
print(model.score(X_valid, y_valid))
#print(accuracy_score(y_valid, y_predict))


# # Voting Classifier

# In[59]:


# Best Result in the voting Classifier 
model = VotingClassifier([
    ('nb', GaussianNB()),
    ('knn', KNeighborsClassifier(5)),
    ('svm', SVC(kernel='linear', C=0.1)),
    ('tree1', DecisionTreeClassifier(max_depth=4)),
    ('tree2', DecisionTreeClassifier(min_samples_leaf=10)),
])
model.fit(X_train, y_train)
y_predict = model.predict(X_valid)
print("For GPS left walk")
print(model.score(X_train, y_train))
#print(model.score(X_valid, y_valid))
print(accuracy_score(y_valid, y_predict))


# # Apply Machine learning tools on 2nd Dataset

# In[60]:


X = acc_read_variable2[['time','ax', 'ay','az']]
y = acc_read_variable2['Speed (m/s)']

lab_enc = preprocessing.LabelEncoder()
X.ax = lab_enc.fit_transform(X.ax)
X.ay = lab_enc.fit_transform(X.ay)
X.az = lab_enc.fit_transform(X.az)
X.time = lab_enc.fit_transform(X.time)


y = lab_enc.fit_transform(y)
X_train, X_valid,y_train, y_valid = train_test_split(X,y)

#Random Forest Classifier 

model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10)                  
model.fit(X_train,y_train)
model.score(X_valid, y_valid)
y_predict = model.predict(X_valid)
#test_result = rfc.predict(read_variable_test)
#print(result)

print("For GPS Right walk")
print(model.fit(X_train,y_train))
#print(model.score(X_valid, y_valid))
print(accuracy_score(y_valid, y_predict))


# # Gaussian

# In[61]:


# Gaussian 
model = GaussianNB()
model.fit(X_train, y_train)
y_predict = model.predict(X_valid)
print("For GPS right walk")
print(model.score(X_train, y_train))
#print(model.score(X_valid, y_valid))

print(accuracy_score(y_valid, y_predict))


# # K-nearest neighbours classifier.

# In[62]:


# K-nearest neighbours classifier.
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_predict = model.predict(X_valid)
print("For GPS right walk")
print(model.score(X_train, y_train))
#print(model.score(X_valid, y_valid))
print(accuracy_score(y_valid, y_predict))


# # Standard Scaler , KNeighborsClassifier

# In[63]:


# Standard Scaler , KNeighborsClassifier
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=9)
)
model.fit(X_train, y_train)
y_predict = model.predict(X_valid)
print("For GPS right walk")
print(model.score(X_train, y_train))
#print(model.score(X_valid, y_valid))
print(accuracy_score(y_valid, y_predict))


# # Neural network regressor

# In[64]:


# neural network regressor
model = MLPRegressor(hidden_layer_sizes=(8, 6),
                     activation='logistic', solver='lbfgs')
model.fit(X_train, y_train)
print("For GPS Right walk")
print(model.score(X_train, y_train))

print(model.score(X_valid, y_valid))


# #  RandomForestRegressor

# In[65]:


#  RandomForestRegressor
model = RandomForestRegressor(30, max_depth=4)
model.fit(X_train, y_train)
print("For GPS Right walk")
y_predict = model.predict(X_valid)
print(model.score(X_train, y_train))
print(model.score(X_valid, y_valid))
#print(accuracy_score(y_valid, y_predict))


# # K-Neighbor Regressor

# In[66]:


# K-Neighbor Regressor 
model = KNeighborsRegressor(5)
model.fit(X_train, y_train)
print("For GPS Right walk")
y_predict = model.predict(X_valid)
print(model.score(X_train, y_train))
print(model.score(X_valid, y_valid))
#print(accuracy_score(y_valid, y_predict))


# # Voting Classifier 

# In[67]:


# Best Result in the voting Classifier 
model = VotingClassifier([
    ('nb', GaussianNB()),
    ('knn', KNeighborsClassifier(5)),
    ('svm', SVC(kernel='linear', C=0.1)),
    ('tree1', DecisionTreeClassifier(max_depth=4)),
    ('tree2', DecisionTreeClassifier(min_samples_leaf=10)),
])
model.fit(X_train, y_train)
y_predict = model.predict(X_valid)
print("For GPS Right walk")
print(model.score(X_train, y_train))
#print(model.score(X_valid, y_valid))
print(accuracy_score(y_valid, y_predict))

