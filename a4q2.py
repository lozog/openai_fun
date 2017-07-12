from math import isclose
import numpy as np
from random import random

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

# a 3 sided die that lands on side 0 with probability a,
# side 1 with probability b, and side 2 with probability b
# taken from https://piazza.com/class/j25q3iuqs344qb?cid=379
def chooseAction(a, b):
    rand = random()
    if rand <= a:
        return 0
    elif rand <= a + b:
        return 1
    else:
        return 2

# transition function
# given a (state, action) pair, returns the nextState
def T(state, action):
    if (state == 15 or state == 16):
        return state
    else:
        if (action == 0): # up
            if (0 <= state and state <= 3): # if state in top row
                return state
            else:
                return state - 4

        if (action == 1): # down
            if (12 <= state and state <= 15): # if state in bottom row
                return state
            else:
                return state + 4

        if (action == 2) # left
            if (state % 4 == 0): # if state in left-most column
                return state
            else:
                return state - 1

        if (action == 3): # right
            if (state+1 % 4 == 0): # if state in right-most column
                return state
            else:
                return state + 1

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

        pass
    # while
