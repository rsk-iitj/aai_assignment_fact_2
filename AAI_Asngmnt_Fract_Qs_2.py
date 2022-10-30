import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import string

# Variables
outcomes = ('R', 'G', 'B')
t = 10
no_state = 5

# First thing first, let's create the 5 Jars
def create_jars_ball():
    jars =[]
    for _ in range(5):
        this_jar = [random.choice('RGB') for _ in range(random.randint(20, 30))]
        jars.append(this_jar)
        return jars

JARS = create_jars_ball()

def get_random_observations():
    return [random.choice('RGB') for _ in range(10)]

def get_states():
    states = [x for x in JARS]
    return states

def generate_a():
    a = []
    temp_dict = {}
    for _ in range(no_state):
        for jar in JARS:
            temp_dict[jar]=random.uniform(0, 1)
    a.append(temp_dict)
    return tuple(a)

def generate_b():
   b = dict()
   for _ in range(len(JARS)):



def gen_transition_matrix(transitions, n):
    M = np.zeros((5, 5))

    for (i, j) in zip(transitions, transitions[1:]):
        M[i][j] += 1
    for row in M:
        s = sum(row)

        if s > 0:
            row[:] = [f / s for f in row]
    return M

t = [0, 0, 1, 0, 0, 2, 0, 3, 4, 0, 3, 4, 0, 4, 1, 1, 2, 2, 1, 3, 1, 4, 2, 2, 3, 2, 4, 3, 3, 4, 4, 0]
n = max(t) + 1
print(f"Max no of states: {n}")
m = gen_transition_matrix(t, n)
for row in m: print(' '.join('{0:.2f}'.format(x) for x in row))


B = np.zeros((5, 3))


# print(B)
def count_prob(jar, i):
    total = len(jar)
    red = jar.count('R')
    green = jar.count('G')
    blue = jar.count('B')
    probR = red / total
    probG = green / total
    probb = blue / total
    B[i][0] = probR
    B[i][1] = probG
    B[i][2] = probb
    print(f"Probability :  P(R) :{probR}, P(G) :{probG} , P(B) :{probb}")


count_prob(jar_1, 0)
count_prob(jar_2, 1)
count_prob(jar_3, 2)
count_prob(jar_4, 3)
count_prob(jar_5, 4)
print(B)


#initial State
states =['jar_1','jar_2','jar_3','jar_4','jar_5']
initial_state_pi = random.choice(states)
print(initial_state_pi)



# initial_= [np.random.dirichlet(np.ones(10),size=1)]
initial_distribution = []
initial_ = np.random.dirichlet(np.ones(5), size=1)
for j in range(5):
    x = initial_[0, j]
    initial_distribution.append(round(x, 2))

print(f"PI value: {initial_distribution}")


def forward(O, a, b, pi):
    N = a.shape[0]
    T = 10
    alpha = np.zeros((T, N))

    alpha[0, :] = pi * b[:, O[0]]
    for i in range(N):
        alpha[0, i] = pi[i] * b[i, O[0]]
    for t in range(1, 10):
        for j in range(N):
            dot_prod = alpha[t - 1].dot(a[:, j])
            alpha[t, j] = dot_prod * b[j, O[t]]

    return alpha


alpha = forward(O, m, B, initial_distribution)
print(f"αT(i): {alpha[0, :]}")
sumoflastrow = alpha[0, :].sum()
print("P(O|λ):  {:.2f}".format(round(sumoflastrow, 2)))
