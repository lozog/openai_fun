import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd#,Adam


env = gym.make('CartPole-v0')

EPISODES = 20

discount = 0.99
epsilon = 0.05


model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2))
model.compile(loss='mse', optimizer=sgd(lr=0.01))

for i_episode in range(EPISODES):
    observation = env.reset()
    observation = np.reshape(observation, [1, 4])
    for t in range(5000):
        env.render()
        # print(observation)
        action = model.predict(observation)
        action = np.argmax(action[0])
        if np.random.uniform(0,1) < epsilon:
            # Either 0 or 1 sample the action randomly
            action = np.random.randint(2)
        # action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        observation = np.reshape(observation, [1, 4])
        print(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

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
