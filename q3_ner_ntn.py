import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adagrad
import matplotlib.pyplot as plt
from math import pow

EPISODES = 1000
HORIZON = 500
np.random.seed(10)

class DQN:
    def __init__(self, discount=0.99, epsilon=0.05):
        self.discount = discount
        self.epsilon = epsilon

        self.model = Sequential()
        self.model.add(Dense(10, input_dim=4, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(2))
        self.model.compile(loss='mse', optimizer=Adagrad(lr=0.1))

    def plot(self, totalDiscountedRewards):
        plt.plot(range(1, EPISODES + 1), totalDiscountedRewards)
        plt.ylabel('Total Discounted Reward')
        plt.xlabel('# of Training Episodes')
        plt.grid()
        axes = plt.gca()
        plt.title("No Experience Replay, No Target Network")
        plt.show()

    def train(self, observation, prevObservation, prevPrediction, reward, action, done, t):
        target = reward
        if (not done):
            prediction = self.model.predict(observation)
            # print("{}, {}".format(prediction[0], np.amax(prediction[0])))
            target = reward + self.discount * np.amax(prediction[0])

        prevPrediction[0][action] = target
        self.model.fit(prevObservation, prevPrediction, epochs=1, verbose=0)


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
                self.train(observation, prevObservation, prevPrediction, reward, action, done, t)

                if (done):
                    print("Episode {} finished after {} timesteps".format(ep+1, t+1))
                    break
            # for
            totalDiscountedRewards.append(totalDiscountedReward)
        # for

        # print(totalDiscountedRewards)
        self.plot(totalDiscountedRewards)

dqn = DQN()
dqn.run()
