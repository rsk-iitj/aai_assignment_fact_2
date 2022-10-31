import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

# Initial Assumptions
actual_temp = 33.5
err_in_measur = 3.5
err_in_est = 1.4
est_value = 32.5
no_mears = 10

# List DS to store values to display
mesr_value_list = []
est_value_list = []
kalman_gain_list = []
err_est_list = []

# Calculate Kalman Gain
def return_kalman_gain(error_est, error_msr):
    return error_est / (error_est + error_msr)

# Generate randomly 
def return_mesure_temp():
    return random.uniform(30, 35)

def store_data_df(mesur_temp,err_in_est,est_value,kalman_gain):
    err_est_list.append(err_in_est)
    est_value_list.append(est_value)
    kalman_gain_list.append(kalman_gain)
    mesr_value_list.append(mesur_temp)

def calculate_est_temp(err_in_est,est_value):
    for i in range(no_mears):
        mesur_temp = return_mesure_temp()
        kalman_gain = return_kalman_gain(err_in_est, err_in_measur)
        est_value = est_value + \
            kalman_gain * (mesur_temp - est_value)
        err_in_est = (1 - kalman_gain) * err_in_est
        store_data_df(mesur_temp, err_in_est, est_value, kalman_gain)
    return est_value

def print_plot_results(final_est_temp):
    df_mesr = pd.DataFrame({'Measured Temp':mesr_value_list, 'Kalman Gain':kalman_gain_list,'Estimated Temp':est_value_list,'Error In Estimate':err_est_list})
    print(df_mesr.to_string())
    print(f"\nFinal Estimated Temperature : {est_value:.2f}")
    plt.plot( np.linspace(0, 100, no_mears), np.full(no_mears, actual_temp), label="Actual Value")
    plt.plot( np.linspace(0, 100, no_mears), mesr_value_list, label="Measured Value")
    plt.plot( np.linspace(0, 100, no_mears), est_value_list, label="Estimated Value")
    plt.title("Trend of Measured,Estimated & Correct Temperature")
    plt.xlabel("No of Iterations - Time")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()

final_est_temp = calculate_est_temp(err_in_est,est_value)
print_plot_results(final_est_temp)
