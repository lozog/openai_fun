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

    # NUM_ITERATIONS = 1 # how many times to run the value iteration process
    NUM_STATES = 17 # includes goal state
    MAX_ERROR = 0.01 # defined by assignment
    ACTIONS = ['up', 'down', 'left', 'right', 'none']

    # VF = U, nextVF = U'
    nextVF = np.zeros(NUM_STATES)

    # policy will include an action for each state.
    # initial values won't affect final outcome because value iteration converges to optimal policy
    # policy = np.zeros(NUM_STATES, dtype=np.int)
    policy = {key: '' for key in range(0, NUM_STATES)}

    delta = 0 # maximum change in the utility of any state in an iteration
    iter_count = 0
    while (True):
        VF = np.copy(nextVF)
        # print("VF: {}".format(VF))

        error = np.zeros(NUM_STATES)

        for state in range(NUM_STATES):

            maxExpectedUtility = 0 # expected utility, assuming we choose optimal action
            bestMove = 4           # action 4 is none, indicating the agent's action doesn't change the outcome

            for action in range(0,4):

                expectedUtility = 0
                # print(action)

                for nextState in range(NUM_STATES):
                    expectedUtility += T[state][nextState][action] * nextVF[nextState]
                # for

                if (expectedUtility > maxExpectedUtility):
                    maxExpectedUtility = expectedUtility
                    bestMove = int(action)
            # for

            # calculate utility
            reward = R[state]
            nextVF[state] = reward + (discount * maxExpectedUtility)
            policy[state] = ACTIONS[bestMove]

            # calculate error
            error[state] = abs(nextVF[state] - VF[state])
            # print("abs({} - {} = {})".format(nextVF[state], VF[state], error[state]))

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
    return nextVF,policy


pol1,v1 = valueIteration(0.99, 0.9, 0.05)
pol2,v2 = valueIteration(0.99, 0.8, 0.1)

print("0.9, 0.05: {}\n{}".format(pol1, v1))
print("0.8, 0.1: {}\n{}".format(pol2, v2))
