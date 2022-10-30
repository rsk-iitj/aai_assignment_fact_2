import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import string

T = 10
no_state = 5

JAR_1 = [random.choice(['Red','Green','Blue']) for _ in range(random.randint(20, 30))]
JAR_2 = [random.choice(['Red','Green','Blue']) for _ in range(random.randint(20, 30))]
JAR_3 = [random.choice(['Red','Green','Blue']) for _ in range(random.randint(20, 30))]
JAR_4 = [random.choice(['Red','Green','Blue']) for _ in range(random.randint(20, 30))]
JAR_5 = [random.choice(['Red','Green','Blue']) for _ in range(random.randint(20, 30))]

JARS = [JAR_1,JAR_2,JAR_3,JAR_4,JAR_5]

B = np.zeros((5, 3))

def get_balls_proabbility_jars(jar):
    total_balls = len(jar)
    red_balls = jar.count('Red')
    green_balls = jar.count('Green')
    blue_balls = jar.count('Blue')
    red_prob = red_balls / total_balls
    green_prob = green_balls / total_balls
    blue_prob = blue_balls / total_balls
    for i in range(5):
        for j in range(3):
            if j == 0:
                B[i][j] = red_prob
            elif j == 1:
                B[i][j] = green_prob
            elif j == 2:
                B[i][j] = blue_prob


def get_B():
    for jar in JARS:
        get_balls_proabbility_jars(jar)

def get_initial_state():
    states =['JAR_1','JAR_2','JAR_3','JAR_4','JAR_5']
    init_state = random.choice(states)
    print(f"Initial State is :{init_state}")

def generate_observations():
    O = [random.randrange(0, 3) for _ in range(10)]
    return O

def return_transition_matrix(tans):
    matrix = np.zeros((no_state, no_state))
    for (row, col) in zip(tans, tans[1:]):
        matrix[row][col] += 1
    for this_row in matrix:
        this_sum = sum(this_row)
        if this_sum > 0:
            this_row[:] = [p / this_sum for p in this_row]
    return matrix

def generate_pi():
    pi = []
    int_vals = np.random.dirichlet(np.ones(5), size=1)
    for i in range(5):
        init = int_vals[0, i]
        pi.append(round(init, 2))
    return pi


def forward(O, a, b, pi):
    N = a.shape[0]
    alpha = np.zeros((T, N))
    alpha[0, :] = pi * b[:, O[0]]
    for n in range(N):
        alpha[0, n] = pi[n] * b[n, O[0]]
    for t in range(1, T):
        for n in range(N):
            product = alpha[t - 1].dot(a[:, n])
            alpha[t, n] = product * b[n, O[t]]
    return alpha

get_initial_state()
get_B()
O = generate_observations()
pi = generate_pi()
transitions = [0,0,1,0,0,2,0,3,4,0,3,4,0,4,1,1,2,2,1,3,1,2,2,3,3,2,2,1,2,4,3,0]
M = return_transition_matrix(transitions)
alpha = forward(O, M, B, pi)
print(f"alphaT(i): {alpha[0, :]}")
prob_given_lambda_obs = alpha[0, :].sum()
print("P(O|λ):  {:.2f}".format(round(prob_given_lambda_obs, 2)))