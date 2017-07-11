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

def valueIteration(discount, a = 0.8, b = 0.1):
    if ( not isclose(a + b + b, 1.0) ): # fricken floats
        print("Error: a + b + b must equal 1")
        return 1

    T,R = gridWorld(a, b)

    # policy = U, nextPolicy = U'
    nextPolicy = np.zeros(17) # policy will include an action for each state.
                              # initial values won't affect final outcome because value iteration converges to optimal policy

    nextV = np.zeros(17)

    # NUM_ITERATIONS = 1 # how many times to run the value iteration process
    NUM_STATES = 17 # includes goal state
    MAX_ERROR = 0.01 # defined by assignment

    delta = 0 # maximum change in the utility of any state in an iteration
    iter_count = 0
    while (True):
        policy = np.copy(nextPolicy)
        V = np.copy(nextV)
        # print("policy: {}".format(policy))

        error = np.zeros(17)

        for state in range(NUM_STATES):

            maxExpectedUtility = 0 # expected utility, assuming we choose optimal action
            bestMove = -1          # -1 is a sentinel value

            for action in range(0,4):

                expectedUtility = 0
                # print(action)

                for nextState in range(NUM_STATES):
                    expectedUtility += T[state][nextState][action] * nextPolicy[nextState]
                # for

                if (expectedUtility > maxExpectedUtility):
                    maxExpectedUtility = expectedUtility
                    bestMove = action

            # for

            # calculate utility
            reward = R[state]
            nextPolicy[state] = reward + (discount * maxExpectedUtility)
            nextV[state] = bestMove

            # calculate error
            error[state] = abs(nextPolicy[state] - policy[state])
            # print("abs({} - {} = {})".format(nextPolicy[state], policy[state], error[state]))


        # for
        iter_count += 1

        # print("error: {}".format(error))
        isConverged = True # with apologies to Peter Buhr
        for entry in error:
            if (entry > 0.01):
                isConverged = False
                break
        # for

        if (isConverged):
            break

    # while

    print("{} iterations".format(iter_count))
    return nextPolicy,nextV

pol1,v1 = valueIteration(0.99, 0.9, 0.05)
pol2,v2 = valueIteration(0.99, 0.8, 0.1)
print("0.9, 0.1: {}\n{}".format(pol1, v1))
print("0.8, 0.05: {}\n{}".format(pol2, v2))
