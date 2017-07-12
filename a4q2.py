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
# possible actions: up (0), down (1), left (2), right (3)

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


def QLearning(discount, a = 0.8, b = 0.1):
    if ( not isclose(a + b + b, 1.0) ): # fricken floats
        print("Error: a + b + b must equal 1")
        return 1

    # NUM_ITERATIONS = 1 # how many times to run the value iteration process
    NUM_STATES = 17 # includes goal state
    ACTIONS = {0:'up', 1:'down', 2:'left', 3:'right', 4:'none'}
    START_STATE = 4
    BAD_STATE   = 9
    GOAL_STATE  = 15
    END_STATE   = 16



    # initialize rewards to -1
    R = -np.ones((17))

    # set rewards
    R[15] = 100;  # goal state
    R[9] = -70;   # bad state
    R[16] = 0;    # end state

    # initialize Q-values to 0
    Q = np.zeros(NUM_STATES, 4)

    state = START_STATE

    while (True):

        # select action a and execute it


    # while
