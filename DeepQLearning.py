import numpy as np
import pandas as pd
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from sklearn.preprocessing import LabelEncoder

# set parameters
gamma = 0.95 # discount rate
epsilon = 1.0 # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory = deque(maxlen=2000)

# load dataset
dataset = pd.read_csv(r'Fantasy Football Dataset - Sheet1.csv')
X = dataset.iloc[:, :-5].values
y = dataset.iloc[:, -4].values

#Label Encoding
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X[:, 2] = le.fit_transform(X[:, 2])
X[:, 3] = le.fit_transform(X[:, 3])

# define the Deep Q-Learning Network model
def DQN_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model

# define the agent
class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = memory
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQN_model(state_size, action_size)

    # get action based on epsilon-greedy policy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = state.astype('float32')
            return np.argmax(self.model.predict(state)[0])

    # store experience in memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # replay experience and update Q values
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            state = state.astype(float)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# initialize the agent and the environment
state_size = X.shape[1]
action_size = 1 # in this case, the action is to predict the best player
agent = DQLAgent(state_size, action_size)

# train the agent
for e in range(100): #Original set to 500
    state = X[1]
    state = np.reshape(state, [1, state_size])
    done = False
    while not done:
        action = agent.act(state)
        next_state = X[action]
        next_state = np.reshape(next_state, [1, state_size])
        reward = y[action]
        done = True # this is a simplified environment with only one step
        agent.remember(state, action, reward, next_state, done)
        state = next_state
    agent.replay(batch_size)

# predict the best player
X = np.asarray(X).astype(np.float32)
best_player_index = np.argmax(agent.model.predict(X)[0])
best_player = X[best_player_index]
dataset = np.asarray(dataset)
for i in range(len(best_player)):
    print()
    print("The best player is: ", dataset[i],": ",  best_player[i])
