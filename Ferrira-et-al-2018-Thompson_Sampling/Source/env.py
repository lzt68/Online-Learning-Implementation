import numpy as np
from numpy.random import Generator, PCG64


class Environment_Bernoulli:
    # the demand follows Bernoulli Distribution
    def __init__(self, theta, K, N, random_seed=12345):
        self.theta = theta
        self.K = K
        self.N = N
        self.random_seed = random_seed

        self.random_generator = Generator(PCG64(random_seed))

    def response(self, price_offered_index):
        # vectorize binomial sample function to accelerate
        mybinomial = np.vectorize(self.random_generator.binomial)
        if price_offered_index < self.K + 1:
            demand = mybinomial(np.ones(self.N), self.theta[price_offered_index - 1, :])  # record the realization of demand
        else:  # the demand must be zero
            demand = np.zeros(self.N)
        return demand


#%% unit test 1
# from numpy.random import Generator, PCG64

# rng1 = Generator(PCG64(12345))
# rng2 = Generator(PCG64(0))

# print("One by one")
# print(rng1.standard_normal())
# print(rng2.standard_normal())
# print(rng1.standard_normal())
# print(rng2.standard_normal())

# rng1 = Generator(PCG64(12345))
# rng2 = Generator(PCG64(0))

# np.random.seed(5)
# print("In a row")
# print(rng1.standard_normal())
# print(rng1.standard_normal())
# print(rng2.standard_normal())
# print(rng2.standard_normal())

#%% unit test 2
# from numpy.random import Generator, PCG64

# rng1 = Generator(PCG64(12345))
# rng2 = Generator(PCG64(0))

# print("One by one")
# print(rng1.binomial(1.0, 0.5))
# print(rng2.binomial(1.0, 0.5))
# print(rng1.binomial(1.0, 0.5))
# print(rng2.binomial(1.0, 0.5))

# rng1 = Generator(PCG64(12345))
# rng2 = Generator(PCG64(0))

# print("In a row")
# print(rng1.binomial(1.0, 0.5))
# print(rng1.binomial(1.0, 0.5))
# print(rng2.binomial(1.0, 0.5))
# print(rng2.binomial(1.0, 0.5))
