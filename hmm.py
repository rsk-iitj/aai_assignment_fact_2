import numpy as np


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


def calc_POL(A: dict[str, dict[str, float]], B: dict[str, dict[str, float]], pi: dict[str, float], observations: list[str]):
    states =  list(A.keys())
    N = len(states)
    outcomes = list(list(B.values())[0].keys())
    print(outcomes)
    M = len(outcomes)
    T = len(observations)
    O = list(map(lambda x: outcomes.index(x) , observations))
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


# no of states
n = 5
states = ('jar_1', 'jar_2', 'jar_3', 'jar_4', 'jar_5',)
# no of diff observation outcome
m = 3
outcomes = ('red', 'green', 'blue')
# length of observation sequence
t = 10

a1 = (
    {'jar_1': 0.2, 'jar_2': 0.3, 'jar_3': 0.13, 'jar_4': 0.07, 'jar_5': 0.3})
a2 = (
    {'jar_1': 0.2, 'jar_2': 0.3, 'jar_3': 0.13, 'jar_4': 0.07, 'jar_5': 0.3})
a3 = (
    {'jar_1': 0.2, 'jar_2': 0.3, 'jar_3': 0.13, 'jar_4': 0.07, 'jar_5': 0.3})
a4 = (
    {'jar_1': 0.2, 'jar_2': 0.3, 'jar_3': 0.13, 'jar_4': 0.07, 'jar_5': 0.3})
a5 = (
    {'jar_1': 0.2, 'jar_2': 0.3, 'jar_3': 0.13, 'jar_4': 0.07, 'jar_5': 0.3})

b1 = ({'red': 0.1, 'green': 0.4, 'blue': 0.5})
b2 = ({'red': 0.9, 'green': 0.04, 'blue': 0.06})
b3 = ({'red': 0.15, 'green': 0.45, 'blue': 0.40})
b4 = ({'red': 0.5, 'green': 0.4, 'blue': 0.1})
b5 = ({'red': 0.3, 'green': 0.4, 'blue': 0.3})

A = (
    {'jar_1': a1, 'jar_2': a2, 'jar_3': a3, 'jar_4': a4, 'jar_5': a5})
B = (
    {'jar_1': b1, 'jar_2': b2, 'jar_3': b3, 'jar_4': b4, 'jar_5': b5})
pi = (
    {'jar_1': 0.2, 'jar_2': 0.3, 'jar_3': 0.13, 'jar_4': 0.07, 'jar_5': 0.3})

lamda = (A, B, pi)
observations = ['red', 'green', 'red', 'red', 'blue', 'blue', 'blue', 'green', 'red','red']
prob_O_given_lamda = calc_POL(*lamda, observations)


print("\nλ is { A, B, pi }")
print(f"\nA is \n{A}")
print(f"\nB is \n{B}")
print(f"\npi is \n{pi}")
print(f"\nP(O|λ) for O {observations} is {prob_O_given_lamda:f}")
