import numpy as np


class Environment:
    # we use an object as environment to cooperate with agents
    def __init__(self, theta, K, N):
        self.theta = theta
        self.K = K
        self.N = N

    def Response(self, price_offered_index):
        # vectorize binomial sample function to accelerate
        mybinomial = np.vectorize(np.random.binomial)
        if price_offered_index < self.K + 1:
            demand = mybinomial(np.ones(self.N), self.theta[price_offered_index - 1, :])  # record the realization of demand
        else:  # the demand must be zero
            demand = np.zeros(self.N)
        return demand
