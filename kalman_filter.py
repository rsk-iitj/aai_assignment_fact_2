# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# %%

Estimated_temp =34 #Est
Error_measurement= 4 #M
error_in_estimate= 2 #Error in Estimation


# %%
Est =[]
Kalman_gain= []
Error_in_estimation=[]
Measure = random.randint(30,35)
iteration =4



# %%
#kalman Gain
def Kalman_gain_func(error_in_estimate_p, Measure_p):
    
    p= error_in_estimate_p + Measure_p
    return (error_in_estimate_p /p)
#Estimation
def Estim_func(Estimated_temp_prev, Measure,kg):
    Estim = Estimated_temp_prev + kg *( Measure- Estimated_temp_prev)
    return Estim   
def Error_calc(kg,Estim_prev):
    error_now = (1-kg)*Estim_prev 
    return error_now
    

# %%

#initial estimations

for i in range(0,10):
    kg =Kalman_gain_func(error_in_estimate,Error_measurement)
    Kalman_gain.append(kg)
    
    
    
    Estimated_temp= Estim_func(Estimated_temp, Measure, kg)
    Est.append(Estimated_temp)
    
    Estim_prev= error_in_estimate
    error_in_estimate =Error_calc(kg,Estim_prev)
    Error_in_estimation.append(error_in_estimate)

# %%
print(Kalman_gain)
print(Est)
print(Error_in_estimation)
print(f"Actual Temperature: {Measure}")

# %%
y1 = Est
y2= [Measure] *10
x= [1,2,3,4,5,6,7,8,9,10]
plt.xlabel('Iterations')
plt.ylabel('Temperature')
plt.plot(x,y1, label= "Estimated_Temp")
plt.plot(x,y2, label= "Measure_Temp")
plt.legend()

plt.show()


