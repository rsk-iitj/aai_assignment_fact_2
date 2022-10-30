import random

import matplotlib.pyplot as plt
import numpy as np


# compute kalman gain
def get_kalman_gain(error_est, error_msr):
    return error_est / (error_est + error_msr)


# assumptions
true_value = 33
error_in_measurement = 4
error_in_estimate = 2
estimate_of_value = 32
no_of_measurement = 10  # 3

# m=[75,71,70]
list_of_measured_value = []
list_of_estimated_value = []

for i in range(no_of_measurement):
    measurred_temp = random.uniform(30, 35)  # m[i]
    kalman_gain = get_kalman_gain(error_in_estimate, error_in_measurement)
    estimate_of_value = estimate_of_value + \
        kalman_gain * (measurred_temp - estimate_of_value)
    error_in_estimate = (1 - kalman_gain) * error_in_estimate
    print([measurred_temp, kalman_gain, estimate_of_value, error_in_estimate])
    list_of_estimated_value.append(estimate_of_value)
    list_of_measured_value.append(measurred_temp)

print(f"\nFinal estimate of tempararue : {estimate_of_value:.2f}")

time_intervals = x = np.linspace(0, 100, no_of_measurement)
fig, ax = plt.subplots()
ax.plot(time_intervals, np.full(no_of_measurement, true_value), label="True Value")
ax.plot(time_intervals, list_of_measured_value, label="Measured Value")
ax.plot(time_intervals, list_of_estimated_value, label="Estimated Value")
ax.legend()
plt.show()
