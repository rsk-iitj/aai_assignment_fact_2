import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import string

# Variables
outcomes = ('R', 'G', 'B')
t = 10
no_state = 5
m = 3

# First thing first, let's create the 5 Jars
def create_jars_ball():
    jars =[]
    for i in range(5):
        this_jar = "jar-"+str(i+1)
        jars.append(this_jar)
    return jars

JARS = create_jars_ball()
print(JARS)

def get_random_observations():
    return [random.choice('RGB') for _ in range(10)]

def get_states():
    states = [str(x) for x in JARS]
    return states

def generate_a():
    a = []
    temp_dict = {}
    for _ in range(no_state):
        for jar in JARS:
            temp_dict[jar]=round(random.uniform(0, .9),2)
        a.append(temp_dict)
    return (a)

def generate_b():
   b = []
   temp_dict = {}
   for _ in range(no_state):
       for color in outcomes:
           temp_dict[color] = round(random.uniform(0, .9),2)
       b.append(temp_dict)
   return (b)


def generate_A():
    A = {}
    a = generate_a()
    print(a)
    count=0
    for jar in JARS:
        A[jar]= a[count]
        count+=1
    #A.append(temp_dict)
    return A


def generate_B():
    B = {}
    #temp_dict = {}
    b = generate_b()
    for i,jar in enumerate(JARS):
        B[jar]= b[i]
    #B.append(temp_dict)
    return B


def generate_pi():
    pi = {}
    #temp_dict = {}
    for i,jar in enumerate(JARS):
        pi[jar]= round(random.uniform(0, .9),2)
    #pi.append(temp_dict)
    return pi

def forward(N, T, O, a, b, pi):
    alpha = np.zeros((T, N))

    for i in range(N):
        alpha[0, i] = pi[i] * b[i][O[0]]

    for t in range(1, T):
        for j in range(N):
            alpha_a = 0
            for i in range(N):
                alpha_a += alpha[t - 1][i]*a[i][j]
            alpha[t, j] = alpha_a * b[j][ O[t]]

    return alpha


def calculate_pol(A, B, pi, obvs):
    states =  list(A.keys())
    N = len(states)
    outcomes = list(list(B.values())[0].keys())
    print(outcomes)
    M = len(outcomes)
    T = len(obvs)
    O = list(map(lambda x: outcomes.index(x) , obvs))
    print(O)
    a = [list(A[x].values()) for x in A]
    print(a)
    b = [list(B[x].values()) for x in B]
    print(b)
    alpha = forward(N, T, O, a, b, list(pi.values()))
    print(alpha)
    pol = 0
    for i in range(N):
        pol += alpha[T-1][i]
    return pol

A=generate_A()
B=generate_B()
pi=generate_pi()
obvs=get_random_observations()
#print(obvs)
prob_O= calculate_pol(A,B,pi,obvs)



print("\nλ is { A, B, pi }")
print(f"\nA is \n{A}")
print(f"\nB is \n{B}")
print(f"\npi is \n{pi}")
print(f"\nP(O|λ) for O {obvs} is {prob_O:f}")






