import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adagrad
import matplotlib.pyplot as plt
from math import pow
from collections import deque

EPISODES = 1000
HORIZON = 500
REPLAY_BUFFER_SIZE = 1000
MINI_BATCH_SIZE = 50
WEIGHT_COPY_FREQ = 2
np.random.seed(10)

class DQN:
    def __init__(self, discount=0.99, epsilon=0.05, replay=False, target=False):
        self.discount = discount
        self.epsilon = epsilon
        self.isReplay = replay
        self.isTarget = target

        if (replay):
            self.replayMemory = deque(maxlen = REPLAY_BUFFER_SIZE)

        self.model = Sequential()
        self.model.add(Dense(10, input_dim=4, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(2))
        self.model.compile(loss='mse', optimizer=Adagrad(lr=0.1))

        if (target):
            self.targetModel = Sequential()
            self.targetModel.add(Dense(10, input_dim=4, activation='relu'))
            self.targetModel.add(Dense(10, activation='relu'))
            self.targetModel.add(Dense(2))
            self.targetModel.compile(loss='mse', optimizer=Adagrad(lr=0.1))

    def plot(self, totalDiscountedRewards):
        plt.plot(range(1, EPISODES + 1), totalDiscountedRewards)
        plt.ylabel('Total Discounted Reward')
        plt.xlabel('# of Training Episodes')
        plt.grid()
        axes = plt.gca()
        replayLabel = "No"
        targetLabel = "No"
        if (self.isReplay):
            replayLabel = "With"
        if (self.isTarget):
            targetLabel = "With"
        plt.title("{} Experience Replay, {} Target Network".format(replayLabel, targetLabel))
        plt.show()

    def train(self, observation, prevObservation, prevPrediction, reward, action, done):
        target = reward
        if (not done):
            if (self.isTarget):
                prediction = self.targetModel.predict(observation)
            else:
                prediction = self.model.predict(observation)
            # print("{}, {}".format(prediction[0], np.amax(prediction[0])))
            target = reward + self.discount * np.amax(prediction[0])

        prevPrediction[0][action] = target
        self.model.fit(prevObservation, prevPrediction, epochs=1, verbose=0)

    def replay(self):
        if (len(self.replayMemory) < MINI_BATCH_SIZE):
            return
        else:
            memoryIndices = np.random.choice(len(self.replayMemory), MINI_BATCH_SIZE)
            for mIdx in memoryIndices:
                memory = self.replayMemory[mIdx]
                prevObservation = memory[0]
                action          = memory[1]
                observation     = memory[2]
                reward          = memory[3]
                done            = memory[4]

                prevPrediction = self.model.predict(prevObservation)
                self.train(observation, prevObservation, prevPrediction, reward, action, done)

    def run(self):
        env = gym.make('CartPole-v1')

        totalDiscountedRewards = []

        for ep in range(EPISODES):
            observation = env.reset()
            observation = np.reshape(observation, [1, 4])

            totalDiscountedReward = 0
            for t in range(HORIZON):
                # env.render()

                prediction = self.model.predict(observation)

                action = np.argmax(prediction[0])
                if np.random.uniform(0,1) < self.epsilon:
                    # epsilon-greediness
                    action = np.random.randint(2)

                prevObservation = observation
                prevPrediction = prediction

                observation, reward, done, info = env.step(action)
                observation = np.reshape(observation, [1, 4])

                totalDiscountedReward += pow(self.discount, t)*reward

                if (self.isReplay):
                    self.replayMemory.append([prevObservation, action, observation, reward, done])

                self.train(observation, prevObservation, prevPrediction, reward, action, done)

                if (done):
                    if (ep % 10 == 0):
                        print("Episode {} finished after {} timesteps".format(ep+1, t+1))
                    break
            # for
            totalDiscountedRewards.append(totalDiscountedReward)

            if (self.isReplay):
                self.replay()

            if (self.isTarget):
                if (ep > 0 and ep % WEIGHT_COPY_FREQ == 0):
                    self.targetModel.set_weights( self.model.get_weights() )
        # for

        # print(totalDiscountedRewards)
        self.plot(totalDiscountedRewards)

dqn = DQN(replay=True,target=True)
dqn.run()
