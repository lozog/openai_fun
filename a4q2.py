from math import isclose
import numpy as np
import random

random.seed(1)

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

# taken from https://piazza.com/class/j25q3iuqs344qb?cid=379
def rollDie(a, b):
    rand = random.random()
    if rand <= a:
        # land on face 1
        return 0
    elif rand <= a + b:
        print("rolled 1")
        # land on face 2
        return 1
    else:
        print("rolled 2")
        # land on face 3
        return 2

# picks an action to take, given the epsilon-greediness parameter
def calcAction(e, optimal):
    rand = random.random()
    if rand <= 1-e:
        return optimal
    else:
        # return random int between 0 and 3
        # print("epsilon-greediness")
        return random.randint(0, 3)

# given an action, returns a list of the move and its lateral moves
def possibleOutcomes(action):
    if (action == 0 or action == 1):
        return [action, 2, 3]
    elif (action == 2 or action == 3):
        return [action, 0, 1]

# transition function
# given a (state, action) pair, returns the nextState
def T(state, action):
    if (state == 15 or state == 16):
        return 16
    else:
        # if state in top row
        if (action == 0): # up
            if (0 <= state and state <= 3):
                return state
            else:
                return state - 4

        if (action == 1): # down
            if (12 <= state and state <= 15): # if state in bottom row
                return state
            else:
                return state + 4

        if (action == 2): # left
            if (state % 4 == 0): # if state in left-most column
                return state
            else:
                return state - 1

        if (action == 3): # right
            if ((state+1) % 4 == 0): # if state in right-most column
                # print ("{}+1 % 4 = {}".format(state, (state+1)%4))
                return state
            else:
                return state + 1

def QLearning(NUM_ITERATIONS = 1, epsilon = 0.05, discount = 0.99, a = 0.9, b = 0.05):
    if ( not isclose(a + b + b, 1.0) ): # fricken floats
        print("Error: a + b + b must equal 1")
        return 1

    NUM_STATES = 17 # includes goal state
    ACTIONS = {0:'up', 1:'down', 2:'left', 3:'right', 4:'none'}
    START_STATE = 4
    GOAL_STATE  = 15
    END_STATE   = 16

    # initialize rewards to -1
    R = -np.ones((17))

    # set rewards
    R[15] = 100;  # goal state
    R[9] = -70;   # bad state
    R[16] = 0;    # end state
    # print(R)

    # initialize Q-values to 0
    Q = np.zeros((NUM_STATES, 4))
    # print (Q)

    # initialize N to 0
    N = np.zeros((NUM_STATES, 4), dtype='int')
    # print(N)

    # initialize policy
    policy = np.zeros(NUM_STATES, dtype='int')

    DEBUG = False

    for i in range(0, NUM_ITERATIONS):
        state = START_STATE

        while (True):
        # for x in range(0, 5):
            if (DEBUG):
                print(state)

            if (state == END_STATE):
                break

            # calculate optimal action
            curMaxQ = None
            optimalAction = 0
            for a in range(0, 4):
                if(DEBUG):
                    print("state: {}, action: {}, T(s, a): {},".format(state, a, T(state, a)))

                if (T(state, a) == state):
                    continue
                if (DEBUG):
                    print("{} > {} ?".format(Q[state][a], curMaxQ))
                if (DEBUG):
                    print("state: {}, action: {}".format(state, a))
                if (DEBUG and 0 ):
                    print(Q[state])

                if (curMaxQ is None or Q[state][a] > curMaxQ):
                    if (DEBUG and 0):
                        print("optimal found")
                    curMaxQ = Q[state][a]
                    optimalAction = a
            # for
            policy[state] = optimalAction
            if (DEBUG and 0):
                print("optimal action: {}".format(optimalAction))

            # epsilon-greediness
            action = calcAction(epsilon, optimalAction)

            action = possibleOutcomes(action)[0] #[[rollDie(a, b)]

            N[state][action] += 1
            nextState = T(state, action)
            if (DEBUG and 0 and (state == 5)):
                print("state: {}, action: {}, nextState: {}, N(s, a): {}".format(state, action, nextState, N[state][action]))

            if (action != optimalAction):
                if (DEBUG and 0):
                    print("state: {}, action: {}, nextState: {}, N(s, a): {}".format(state, action, nextState, N[state][action]))
                    print("didn't take optimal path!")
            # receive immediate reward r
            reward = R[state]

            # calculate max Q value of next state
            nextMaxQ = None
            for a in range(0, 4):
                # print(nextState)
                # print(Q[nextState])
                if (nextMaxQ is None or Q[nextState][a] > nextMaxQ):
                    nextMaxQ = Q[nextState][a]
            # for

            learnRate = 1/N[state][action]
            # update Q(a, s)
            Qvalue = Q[state][action]
            Q[state][action] = Qvalue \
                             + learnRate*(reward + discount*(nextMaxQ - Qvalue))
            if (DEBUG and 0 and (state == 5)):
                print("{} + {}*({}+{}*({} - {})) = {}\n".format(Qvalue, learnRate, reward, discount, nextMaxQ, Qvalue, Q[state][action]))
            # if (state == nextState):
            #     print("staying still!")
            # else:
            #     print("XXXXXXXXX")
            state = nextState
            if (DEBUG):
                print(Q)
            if (DEBUG):
                print("")
        # while
        # print(policy)
    # for
    return policy, Q

policy, Q = QLearning(10000)
# print(str(policy).replace("0", "up").replace("1", "down").replace("2", "left").replace("3", "right"))
# print(policy)
for i in range(4):
    print(policy[i*4:(i*4)+4])
for key,Qvalues in enumerate(Q):
    print("{}: {}".format(key, Qvalues))
# print(Q)

# ['Right', 'Right', 'Down', 'Down',
#   'Up',       'Up', 'Right', 'Down',
#   'Left', 'Right', 'Right', 'Down',
#   'Right', 'Right', 'Right', 'Up',
#   'Up']
[[-6.4351627842063435, -6.48184760533724, -6.4405195475559731, -6.4335109189868813],
[-5.8064495012150728, -6.2918115129322487, -6.8318159320389489, -5.5380091986862139],
[-5.1738653076108028, -4.5178391062970693, -5.9649432021389899, -4.5182191883581844],
[-4.4065125537081853, -3.5179718403746372, -4.8955105297663035, -4.138439099207404],
[-7.0796021505053615, -7.2945203260612335, -7.0840123838949269, -7.0798205099083749],
[-6.29931630353299, -65.235175983363419, -13.198230763790601, -6.8750707339974761],
[-5.0383120333601932, -3.4705289856620225, -6.2323175928054404, -3.4696709150046336],
[-4.2973507251335947, -2.3713774975302226, -4.3323703513648502, -3.2317648955973479],
[-10.016495940778368, -8.5081953721293324, -8.3195953793941246, -67.402289450180206],
[-74.388030685626944, -73.56023943514279, -74.119114745418841, -73.170328317650998],
[-7.8561026762874917, -3.5495759680901013, -62.511600373507683, -2.3073038604173668],
[-3.2777126459620245, -1.1931003526285615, -3.2212983604621086, -2.2395102642020257],
[-6.5280529667393301, -6.4931239473502371, -6.4955326375234215, -6.3763244684508438],
[-63.778984282115616, -6.9544687243154097, -7.7351092575557265, -4.6920864267901008],
[-2.5082035378428698, -1.9493358418791469, -5.3519092988406696, -1.1689606465062101],
[0, 0, 0, 0],
[0, 0, 0, 0]]
