import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adagrad
import matplotlib.pyplot as plt
from math import pow

EPISODES = 1000

discount = 0.99
epsilon = 0.1

model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2))
model.compile(loss='mse', optimizer=Adagrad(lr=0.1))

env = gym.make('CartPole-v0')

totalDiscountedRewards = []
np.random.seed(10)

for i_episode in range(EPISODES):
    observation = env.reset()
    observation = np.reshape(observation, [1, 4])

    totalDiscountedReward = 0
    for t in range(500):
        # env.render()

        actions = model.predict(observation)

        action = np.argmax(actions[0])
        if np.random.uniform(0,1) < epsilon:
            # Either 0 or 1 sample the action randomly
            action = np.random.randint(2)

        prevObservation = observation

        observation, reward, done, info = env.step(action)
        observation = np.reshape(observation, [1, 4])

        # learn the model
        target = reward
        if (not done):
            temp = model.predict(observation)
            # print("{}, {}".format(temp[0], np.amax(temp[0])))
            target = reward + discount * np.amax(temp[0])
            totalDiscountedReward += pow(discount, t)*reward

        # print(target)
        # print(prevObservation)
        actions[0][action] = target
        model.fit(prevObservation, actions, verbose=0)

        if done:
            print("Episode {} finished after {} timesteps".format(i_episode+1, t+1))
            break
            # for
    totalDiscountedRewards.append(totalDiscountedReward)
# for

# print(totalDiscountedRewards)
plt.plot(range(1, EPISODES + 1), totalDiscountedRewards)
# plt.plot(range(1, 21), testingAccuracy)
plt.ylabel('Total Discounted Reward')
plt.xlabel('# of Training Episodes')
# plt.xticks(range(0, 21, 5))
# plt.yticks(range(0, 121, 20))
plt.grid()
axes = plt.gca()
# axes.set_xlim([0,20])
# axes.set_ylim([0,120])
plt.show()

# Construct a deep Q-network with the following configuration:
    # Input layer of 4 nodes (corresponding to the 4 state features)
    # Two hidden layers of 10 rectified linear units (fully connected)
    # Output layer of 2 identity units (fully connected) that compute the Q-values of the two actions
# Train this neural network by gradient Q-learning with the following parameter:
    # Discount factor: gamma=0.99
    # Exploration strategy: epsilon-greedy with epsilon=0.05
    # Use the adagradOptimizer(learingRate=0.1), AdamOptimizer(learningRate=0.1) or GradientDescentOptimizer(learningRate=0.01).  The Adagrad and Adam optimizers automatically adjust the learning rate in gradient descent and therefore perform better in practice.
    # Maximum horizon of 500 steps per episode (An episode may terminate earlier if the pole falls before 500 steps.  The gym simulator will set the flag "done" to true when the pole has fallen.)
    # Train for a maximum of 1000 episodes
# Produce a graph that shows the discounted total reward (y-axis) earned in each training episode as a function of the number of training episodes.  Produce 4 curves for the following 4 scenarios:
    # Q-learning (no experience replay and no target network)
    # Q-learning with experience replay (no target network).  Use a replay buffer of size 1000 and replay a mini-batch of size 50 after each new experience.
    # Q-learning with a target network (no experience replay).  Update the the target network after every 2 episodes.
    # Q-learning with experience replay and a target network.  Use a replay buffer of size 1000 and replay a mini-batch of size 50 after each new experience. Update the the target network after every 2 episodes.
