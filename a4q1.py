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

# given a state, returns a dict of successor states for each of the 4 actions
def nextState(state):
    if (state == 15 or state == 16):
        return {"up":16, "down":16, "right":16, "left":16}
    else:
        if (0 <= state and state <= 3):
            upState = state
        else:
            upState = state - 4

        if (12 <= state and state <= 15):
            downState = state
        else:
            downState = state + 4

        if (state+1 % 4 == 0):
            rightState = state
        else:
            rightState = state + 1

        if (state % 4 == 0):
            leftState = state
        else:
            leftState = state - 1

        return {"up":upState, "down":downState, "right":rightState, "left":leftState}

def valueIteration(discount, a, b):
    # print(a + b + b)
    if ( not isclose(a + b + b, 1.0) ): # fricken floats
        print("Error: a + b + b must equal 1")
        return 1

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
#
# for i in range(17):
#     print(nextState(i))
