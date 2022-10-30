import numpy as np
import pandas as pd
from itertools import product
from functools import reduce


class ProbabilityVector:
    def __init__(self, probabilities: dict):
        states = probabilities.keys()
        probs = probabilities.values()

        assert len(states) == len(
            probs), "The probabilities must match the states."
        assert len(states) == len(set(states)), "The states must be unique."
        assert abs(sum(probs) - 1.0) < 1e-12, "Probabilities must sum up to 1."
        assert len(list(filter(lambda x: 0 <= x <= 1, probs))) == len(probs), \
            "Probabilities must be numbers from [0, 1] interval."

        self.states = sorted(probabilities)
        self.values = np.array(list(map(lambda x:
                                        probabilities[x], self.states))).reshape(1, -1)

    @classmethod
    def initialize(cls, states: list):
        size = len(states)
        rand = np.random.rand(size) / (size**2) + 1 / size
        rand /= rand.sum(axis=0)
        return cls(dict(zip(states, rand)))

    @classmethod
    def from_numpy(cls, array: np.ndarray, state: list):
        return cls(dict(zip(state, list(array))))

    @property
    def dict(self):
        return {k: v for k, v in zip(self.states, list(self.values.flatten()))}

    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.states, index=['probability'])

    def __repr__(self):
        return "P({}) = {}.".format(self.states, self.values)

    def __eq__(self, other):
        if not isinstance(other, ProbabilityVector):
            raise NotImplementedError
        if (self.states == other.states) and (self.values == other.values).all():
            return True
        return False

    def __getitem__(self, state: str) -> float:
        if state not in self.states:
            raise ValueError(
                "Requesting unknown probability state from vector.")
        index = self.states.index(state)
        return float(self.values[0, index])

    def __mul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityVector):
            return self.values * other.values
        elif isinstance(other, (int, float)):
            return self.values * other
        else:
            NotImplementedError

    def __rmul__(self, other) -> np.ndarray:
        return self.__mul__(other)

    def __matmul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityMatrix):
            return self.values @ other.values

    def __truediv__(self, number) -> np.ndarray:
        if not isinstance(number, (int, float)):
            raise NotImplementedError
        x = self.values
        return x / number if number != 0 else x / (number + 1e-12)

    def argmax(self):
        index = self.values.argmax()
        return self.states[index]


class ProbabilityMatrix:
    def __init__(self, prob_vec_dict: dict):

        assert len(prob_vec_dict) > 1, \
            "The numebr of input probability vector must be greater than one."
        assert len(set([str(x.states) for x in prob_vec_dict.values()])) == 1, \
            "All internal states of all the vectors must be indentical."
        assert len(prob_vec_dict.keys()) == len(set(prob_vec_dict.keys())), \
            "All observables must be unique."

        self.states = sorted(prob_vec_dict)
        self.observables = prob_vec_dict[self.states[0]].states
        self.values = np.stack([prob_vec_dict[x].values
                                for x in self.states]).squeeze()

    @classmethod
    def initialize(cls, states: list, observables: list):
        size = len(states)
        rand = np.random.rand(size, len(observables)) \
            / (size**2) + 1 / size
        rand /= rand.sum(axis=1).reshape(-1, 1)
        aggr = [dict(zip(observables, rand[i, :])) for i in range(len(states))]
        pvec = [ProbabilityVector(x) for x in aggr]
        return cls(dict(zip(states, pvec)))

    @classmethod
    def from_numpy(cls, array:
                   np.ndarray,
                   states: list,
                   observables: list):
        p_vecs = [ProbabilityVector(dict(zip(observables, x)))
                  for x in array]
        return cls(dict(zip(states, p_vecs)))

    @property
    def dict(self):
        return self.df.to_dict()

    @property
    def df(self):
        return pd.DataFrame(self.values,
                            columns=self.observables, index=self.states)

    def __repr__(self):
        return "PM {} states: {} -> obs: {}.".format(
            self.values.shape, self.states, self.observables)

    def __getitem__(self, observable: str) -> np.ndarray:
        if observable not in self.observables:
            raise ValueError(
                "Requesting unknown probability observable from the matrix.")
        index = self.observables.index(observable)
        return self.values[:, index].reshape(-1, 1)


class HiddenMarkovChain:
    def __init__(self, T, E, pi):
        self.T = T  # transmission matrix A
        self.E = E  # emission matrix B
        self.pi = pi
        self.states = pi.states
        self.observables = E.observables

    def __repr__(self):
        return "HML states: {} -> observables: {}.".format(
            len(self.states), len(self.observables))

    @classmethod
    def initialize(cls, states: list, observables: list):
        T = ProbabilityMatrix.initialize(states, states)
        E = ProbabilityMatrix.initialize(states, observables)
        pi = ProbabilityVector.initialize(states)
        return cls(T, E, pi)

    def _create_all_chains(self, chain_length):
        return list(product(*(self.states,) * chain_length))

    def score(self, observations: list) -> float:
        def mul(x, y): return x * y

        score = 0
        all_chains = self._create_all_chains(len(observations))
        for idx, chain in enumerate(all_chains):
            expanded_chain = list(zip(chain, [self.T.states[0]] + list(chain)))
            expanded_obser = list(zip(observations, chain))

            p_observations = list(
                map(lambda x: self.E.df.loc[x[1], x[0]], expanded_obser))
            p_hidden_state = list(
                map(lambda x: self.T.df.loc[x[1], x[0]], expanded_chain))
            p_hidden_state[0] = self.pi[chain[0]]

            score += reduce(mul, p_observations) * reduce(mul, p_hidden_state)
        return score


# no of states
n = 5
states = ('jar_1', 'jar_2', 'jar_3', 'jar_4', 'jar_5',)
# no of diff observation outcome
m = 3
outcomes = ('red', 'green', 'blue')
# length of observation sequence
t = 10

a1 = ProbabilityVector(
    {'jar_1': 0.2, 'jar_2': 0.3, 'jar_3': 0.13, 'jar_4': 0.07, 'jar_5': 0.3})
a2 = ProbabilityVector(
    {'jar_1': 0.2, 'jar_2': 0.3, 'jar_3': 0.13, 'jar_4': 0.07, 'jar_5': 0.3})
a3 = ProbabilityVector(
    {'jar_1': 0.2, 'jar_2': 0.3, 'jar_3': 0.13, 'jar_4': 0.07, 'jar_5': 0.3})
a4 = ProbabilityVector(
    {'jar_1': 0.2, 'jar_2': 0.3, 'jar_3': 0.13, 'jar_4': 0.07, 'jar_5': 0.3})
a5 = ProbabilityVector(
    {'jar_1': 0.2, 'jar_2': 0.3, 'jar_3': 0.13, 'jar_4': 0.07, 'jar_5': 0.3})

b1 = ProbabilityVector({'red': 0.1, 'green': 0.4, 'blue': 0.5})
b2 = ProbabilityVector({'red': 0.9, 'green': 0.04, 'blue': 0.06})
b3 = ProbabilityVector({'red': 0.15, 'green': 0.45, 'blue': 0.40})
b4 = ProbabilityVector({'red': 0.5, 'green': 0.4, 'blue': 0.1})
b5 = ProbabilityVector({'red': 0.3, 'green': 0.4, 'blue': 0.3})

A = ProbabilityMatrix(
    {'jar_1': a1, 'jar_2': a2, 'jar_3': a3, 'jar_4': a4, 'jar_5': a5})
B = ProbabilityMatrix(
    {'jar_1': b1, 'jar_2': b2, 'jar_3': b3, 'jar_4': b4, 'jar_5': b5})
pi = ProbabilityVector(
    {'jar_1': 0.2, 'jar_2': 0.3, 'jar_3': 0.13, 'jar_4': 0.07, 'jar_5': 0.3})

hmc = HiddenMarkovChain(A, B, pi)
observations = ['red', 'green', 'red', 'red', 'blue', 'blue', 'blue', 'green', 'red','red']

print("\nλ is { A, B, pi }")
print(f"\nA is \n{A.df}")
print(f"\nB is \n{B.df}")
print(f"\npi is \n{pi.df}")
print("\nP(O|λ) for O {} is {:f}".format(
    observations, hmc.score(observations)))
