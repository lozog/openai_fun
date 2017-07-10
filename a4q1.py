from gridWorld import gridWorld
from math import isclose
import numpy as np

# Grid world layout:
#
#  ---------------------
#  |  0 |  1 |  2 |  3 |
#  ---------------------
#  |  4 |  5 |  6 |  7 |
#  ---------------------
#  |  8 |  9 | 10 | 11 |
#  ---------------------
#  | 12 | 13 | 14 | 15 |
#  ---------------------
#
#  Goal state: 15
#  Bad state: 9
#  End state: 16
#
# possible actions: up (0), down (1), right (2), left (3)

def valueIteration(discount):
    T,R = gridWorld()

    # policy = U, newPolicy = U'
    newPolicy = np.zeros(17) # policy will include an action for each state. starts empty, gets filled in

    NUM_ITERATIONS = 1 # how many times to run the value iteration process
    NUM_STATES = 17 # includes goal state

    for iteration in range(NUM_ITERATIONS): # eventually replace with some test to see if we've reached optimal policy
        policy = newPolicy

        for state in range(NUM_STATES):

            maxExpectedUtility = 0 # expected utility, assuming we choose optimal action

            for nextState,transition in enumerate(T[state]):
                expectedUtility = 0

                for prob in transition:
                    expectedUtility += prob * newPolicy[nextState]
                # for

                if (expectedUtility > maxExpectedUtility):
                    maxExpectedUtility = expectedUtility

            # for

            reward = R[state]
            newPolicy[state] = reward + (discount * maxExpectedUtility)

        # for

    #for

    return policy

res = valueIteration(0.99, 0.9, 0.05)

print(res)
