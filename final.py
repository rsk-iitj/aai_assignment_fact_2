import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import string

# Variables
outcomes = ('Red', 'Green', 'Blue')
no_observations = 10
no_states = 5


# First thing first, let's create the 5 Jars
def create_jars_ball():
    jars = {}
    jars['JAR_1'] = [random.choice(['Red', 'Green', 'Blue']) for _ in range(random.randint(20, 30))]
    jars['JAR_2'] = [random.choice(['Red', 'Green', 'Blue']) for _ in range(random.randint(20, 30))]
    jars['JAR_3'] = [random.choice(['Red', 'Green', 'Blue']) for _ in range(random.randint(20, 30))]
    jars['JAR_4'] = [random.choice(['Red', 'Green', 'Blue']) for _ in range(random.randint(20, 30))]
    jars['JAR_5'] = [random.choice(['Red', 'Green', 'Blue']) for _ in range(random.randint(20, 30))]
    return jars


def get_random_observations():
    return [random.choice(['Red', 'Green', 'Blue']) for _ in range(10)]


def get_states():
    states = [str(x) for x in JARS.keys()]
    return states


def generate_a():
    a = []
    temp_dict = {}
    for _ in range(no_states):
        for jar in JARS.keys():
            temp_dict[jar] = round(random.uniform(0, 1), 3)
        a.append(temp_dict)
    return (a)


def get_balls_proabbility_jars(jar, color):
    total_balls = list(JARS.get(jar))
    no_total_balls = len(total_balls)
    no_specific_ball = total_balls.count(color)
    return round(no_specific_ball / no_total_balls, 3)


def generate_b():
    b = []
    temp_dict = {}
    for jar in JARS.keys():
        for color in outcomes:
            temp_dict[color] = get_balls_proabbility_jars(jar, color)
        b.append(temp_dict)
    return (b)


def generate_A():
    A = {}
    a = generate_a()
    count = 0
    for jar in JARS.keys():
        A[jar] = a[count]
        count += 1
    # A.append(temp_dict)
    return A


def generate_B():
    B = {}
    # temp_dict = {}
    b = generate_b()
    for i, jar in enumerate(JARS.keys()):
        B[jar] = b[i]
    # B.append(temp_dict)
    return B


def generate_pi():
    pi = {}
    # temp_dict = {}
    for jar in JARS.keys():
        pi[jar] = round(random.uniform(0, 1), 3)
    # pi.append(temp_dict)
    return pi


def forward(O, a, b, pi):
    alpha = np.zeros((no_observations, no_states))

    for this_state in range(no_states):
        multi = b[this_state][O[0]]
        alpha[0, this_state] = pi[this_state] * multi
    tempvar2=0
    for this_observation in range(1, no_observations):
        for n in range(no_states):
            alpha_a = 0
            for this_state in range(no_states):
                tempvar = a[this_state][n]
                alpha_a += alpha[this_observation - 1][this_state] * tempvar
                tempvar2 = b[n][O[this_observation]]
            alpha[this_observation, n] = alpha_a * tempvar2
    return alpha


def calculate_pol(A, B, pi, observations):
    O = list(map(lambda x: outcomes.index(x), observations))
    pol = 0
    alpha = forward(O, [list(A[x].values()) for x in A], [list(B[x].values()) for x in B], list(pi.values()))
    for this_state in range(no_states):
        pol = pol + alpha[no_observations - 1][this_state]
    return pol

JARS = create_jars_ball()
states = get_states()
print(f"States are {states}")
A = generate_A()
B = generate_B()
pi = generate_pi()
obvs = get_random_observations()
prob_O = calculate_pol(A, B, pi, obvs)
#Because of too many assumption and randomization, forward algorithm might not work so another demo trial.
while prob_O>1 or prob_O<0:
    prob_O = calculate_pol(A,B,pi,obvs)

print("\nλ is { A, B, pi }")
print(f"\nA is \n{A}")
print(f"\nB is \n{B}")
print(f"\npi is \n{pi}")
print(f"\nP(O|λ) for O {obvs} is {prob_O}")






