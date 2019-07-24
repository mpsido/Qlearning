import random
from math import log10, floor
import numpy as np

# Keras DQN
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend as K

# Tensorflow DQN
import tensorflow as tf

# Memory
from collections import deque

def random_argmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return np.random.choice(indices)

def round_to(x, sig_figs):
    if x == 0.0:
        return 0.0
    if x < 0.0:
        return round(x, -int(floor(log10(abs(-x))) - (sig_figs - 1)))
    return round(x, -int(floor(log10(abs(x))) - (sig_figs - 1)))

class Memory():
    def __init__(self, max_size):
        self.len = 0
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        return min(self.len, len(self.buffer))
    
    def add(self, experience):
        self.len += 1
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]

class GamePlayer:

    def __init__(self, env, state_function=lambda x: x):
        self.env = env
        self.state_function = state_function
        self.qtable = {}
        self.max_epsilon = 0.8             # Exploration probability at start
        self.min_epsilon = 0.01            # Minimum exploration probability
        self.randinit = False

    def erase_training(self):
        self.qtable = {}

    def Q(self, state, qtable):
        s = self.state_function(state)
        if s not in qtable:
            qtable[s] = []
            init = 0.0
            for i in range(self.env.action_space.n):
                if self.randinit:
                    init = random.uniform(0, 1)
                qtable[s].append(init)
        return qtable[s]

    @staticmethod
    def epsilon_action(state, env, epsilon, action_function):
        if np.random.rand(1) < epsilon:
            action = env.action_space.sample()
        else:
            action = action_function(state)
        return action

    def epison_q_action(self, state, epsilon):
        return GamePlayer.epsilon_action(state, self.env, epsilon, lambda state: random_argmax(self.Q(state, self.qtable)))

    def epison_double_q_action(self, state, epsilon, Q1, Q2):
        def action_function(self, state):
            Q = [x + y for x, y in zip(self.Q(state, Q1), self.Q(state, Q2))]
            return random_argmax(Q)
        return GamePlayer.epsilon_action(state, self.env, epsilon, lambda state: action_function(self, state))

    def start_game(self, render = False):
        state = self.env.reset()
        if (render):
            self.env.render()
        return state

    def q_trained_action(self, state):
        return random_argmax(self.Q(state, self.qtable))

    def double_trained_action(self, state):
        Q = [x + y for x, y in zip(self.Q(state, self.qtable), self.Q(state, self.Q2))]
        return random_argmax(Q)

    def play_game_step(self, action, render = True):
        new_state, reward, done, info = self.env.step(action)
        if (render):
            self.env.render()
        return new_state, reward, done, info

    def close(self):
        self.end_game()

    def end_game(self):
        self.env.close()

    def train(self, total_episodes, alpha, gamma, epsilon, decay_rate, logEvery=100):
        self.start_game(False)
        # Exploration parameters
        alpha0 = alpha

        reward_list = []
        tot_reward_list = []
        # 2 For life or until learning is stopped
        for episode in range(total_episodes):
            # Reset the environment
            state = self.env.reset()
            done = False
            tot_reward = 0
            while done is False:
                action = self.epison_q_action(state, epsilon)
                if action >= self.env.action_space.n:
                    raise IndexError
                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, done, info = self.play_game_step(action, False)

                # dirty ugly cheat witchcraft that does not even work
                # reward += abs(new_state[0])+10.0*abs(new_state[1])

                # Update Q(s, a):= Q(s, a) + lr [R(s, a) + gamma * max Q(s', a') - Q(s, a)]
                self.Q(state, self.qtable)[action] += alpha * (reward  + gamma * max(self.Q(new_state, self.qtable)) - self.Q(state, self.qtable)[action])

                # Our new state is state
                state = new_state
                tot_reward += reward
            reward_list.append(tot_reward)
            if decay_rate != 0.0:
                epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-decay_rate*episode)

            if logEvery > 0 and (episode+1) % logEvery == 0:
                ave_reward = np.mean(reward_list)
                tot_reward_list.append(ave_reward)
                reward_list = []
                #alpha = alpha0 * (total_episodes - episode)/total_episodes
                print('Episode {} Average Reward: {}, alpha: {}, e: {}, len(Q) {}'.format(episode+1, ave_reward, alpha, epsilon, len(self.qtable)))
        return tot_reward_list

    def double_q_train(self, total_episodes, alpha, gamma, epsilon, decay_rate, logEvery=100):
        self.start_game(False)
        # Exploration parameters
        alpha0 = alpha
        if len(self.qtable) == 0:
            Q1 = {}
            Q2 = {}
        else:
            Q1 = self.qtable
            Q2 = self.Q2

        reward_list = []
        tot_reward_list = []
        # 2 For life or until learning is stopped
        for episode in range(total_episodes):
            # Reset the environment
            state = self.env.reset()
            done = False
            tot_reward = 0
            while done is False:
                action = self.epison_double_q_action(state, epsilon, Q1, Q2)
                if action >= self.env.action_space.n:
                    print(action)
                    print(self.Q(state, Q1))
                    print(self.Q(state, Q2))
                    raise IndexError
                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, done, info = self.play_game_step(action, False)

                tradeoff = random.uniform(0, 1)
                if tradeoff > 0.5:
                    self.Q(state, Q1)[action] += alpha * (reward  + gamma * self.Q(new_state, Q2)[random_argmax(self.Q(state, Q1))] - self.Q(state, Q1)[action])
                else:
                    self.Q(state, Q2)[action] += alpha * (reward  + gamma * self.Q(new_state, Q1)[random_argmax(self.Q(state, Q2))] - self.Q(state, Q2)[action])

                # Our new state is state
                state = new_state
                tot_reward += reward
            reward_list.append(tot_reward)
            if decay_rate != 0.0:
                epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-decay_rate*episode)

            if logEvery > 0 and (episode+1) % logEvery == 0:
                ave_reward = np.mean(reward_list)
                tot_reward_list.append(ave_reward)
                reward_list = []
                #alpha = alpha0 * (total_episodes - episode)/total_episodes
                print('Episode {} Average Reward: {}, alpha: {}, e: {}, len(Q1, Q2) ({}, {})'.format(episode+1, ave_reward, alpha, epsilon, len(Q1), len(Q2)))
        self.qtable = Q1
        self.Q2 = Q2
        return tot_reward_list

    def adversarial_q_train(self, total_episodes, alpha, gamma, epsilon, decay_rate, logEvery=100, adversary_function=None):
        self.start_game(False)
        # Exploration parameters
        alpha0 = alpha
        if self.qtable is None:
            self.qtable = {}
        reward_list = []
        tot_reward_list = []
        # 2 For life or until learning is stopped
        for episode in range(total_episodes):
            # Reset the environment
            state = self.env.reset()
            state = self.state_function(state)
            new_state = ()
            previous_state = state
            previous_action = -1
            done = False
            tot_reward = [0, 0]
            nstep = 0
            while done is False:
                if adversary_function is None:
                    action = self.epison_q_action(state, epsilon)
                else:
                    if (episode+nstep+1)%2 == 1:
                        action = adversary_function(state)
                    else:
                        action = self.epison_q_action(state, epsilon)
                if action >= self.env.action_space.n:
                    raise IndexError
                try:
                    # Take the action (a) and observe the outcome state(s') and reward (r)
                    new_state, reward, done, info = self.env.play_symbol(action, (nstep%2)+1)
                    new_state = self.state_function(new_state)
                except:
                    self.Q(state, self.qtable)[action] = -2
                    tot_reward[(episode+nstep)%2] -= 2
                    tot_reward[(episode+nstep+1)%2] -= 2
                    if (episode+nstep+1)%2 == 1:
                        tot_reward[1] -2
                    else:
                        tot_reward[0] += -2
                    continue

                if done:
                    if (episode+nstep+1)%2 == 1:
                        tot_reward[0] -= reward
                        tot_reward[1] += reward
                    else:
                        tot_reward[0] += reward
                        tot_reward[1] -= reward
                    self.Q(previous_state, self.qtable)[previous_action] = -reward
                    self.Q(state, self.qtable)[action] = reward
                else:
                    self.Q(previous_state, self.qtable)[previous_action] += alpha * (-reward  - gamma * max(self.Q(new_state, self.qtable)) - self.Q(state, self.qtable)[action])
                    self.Q(state, self.qtable)[action] += alpha * (reward  - gamma * max(self.Q(new_state, self.qtable)) - self.Q(state, self.qtable)[action])
                    if (episode+nstep+1)%2 == 1:
                        tot_reward[0] -= reward
                        tot_reward[1] += reward
                    else:
                        tot_reward[0] += reward
                        tot_reward[1] -= reward

                if new_state == state or previous_state == new_state:
                    raise IndexError("What is going on ?", new_state, state, previous_state, action, done)

                nstep += 1

                # Our new state is state
                previous_state = state
                state = new_state
                previous_action = action
            reward_list.append(tot_reward[0])
            if decay_rate != 0.0:
                epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-decay_rate*episode)

            if logEvery > 0 and (episode+1) % logEvery == 0:
                ave_reward = np.mean(reward_list)
                tot_reward_list.append(ave_reward)
                reward_list = []
                #alpha = alpha0 * (total_episodes - episode)/total_episodes
                print('Episode {} Average Reward: {}, alpha: {}, e: {}, len(Q) {}'.format(episode+1, ave_reward, alpha, epsilon, len(self.qtable)))
        return tot_reward_list

    @staticmethod
    def model_fit_memory(state_size, sample_size, gamma, reward_when_done, memory, model, vModel=None):
        Y = []
        batch = memory.sample(sample_size)
        S = np.array([np.array(each[0]).reshape(1, state_size) for each in batch])
        for i, (state, action, reward, done, next_state) in enumerate(batch):
            Y.append(model.predict(np.array(state).reshape(1, state_size))[0])
            next_state = np.array(next_state).reshape(1, state_size)
            if done:
                if reward_when_done is not None:
                    Y[i][action] = reward_when_done
                else:
                    Y[i][action] = reward
            else:
                Qnext = model.predict(next_state)[0]
                Y[i][action] = reward + gamma * np.max(Qnext)
        Y = np.stack(Y, axis=0)
        model.fit(S.reshape(sample_size, state_size), Y, epochs=1, verbose=0)

        if vModel is not None:
            NS = np.array([each[4] for each in batch]).reshape(sample_size, state_size)
            # R = np.array([np.array(each[3]).reshape(1, state_size) for each in batch])
            # NSR = np.concatenate((NS, R), axis=1)
            A = np.array([each[1] for each in batch]).reshape(sample_size, 1, 1)
            SA = np.concatenate((S, A), axis=2)
            vModel.fit(SA.reshape(sample_size, state_size+1), NS.reshape(sample_size, state_size), epochs=1, verbose=0)


    @staticmethod
    def play_episode(env, action_function, memory=None):
        state = env.reset()
        done = False
        tot_reward = 0
        nstep = 0
        while done is False:
            action = action_function(state)
            next_state, reward, done, _ = env.step(action)
            if memory is not None:
                memory.add((state, action, reward, done, next_state))
            state = next_state
            tot_reward += reward
            nstep += 1
        return tot_reward, nstep

    def model_train(self, total_episodes, train_function, layers_size=[24, 24], gamma=0.9, alpha=0.001, reward_when_done=None, logEvery=1, trainVmodel=False):
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        if not hasattr(self, 'memory'):
            self.memory = Memory(200000)
        if not hasattr(self, 'model'):
            print("Creating model")
            nb_layers = len(layers_size)
            if int(nb_layers) < 1:
                raise RangeError(nb_layers, len(layers_size))
            self.model = Sequential()
            for i in range(nb_layers):
                self.model.add(Dense(layers_size[i], input_dim=state_size, activation='relu'))
            self.model.add(Dense(action_size, activation='linear'))
            self.model.compile(loss='mse', optimizer=Adam(lr=alpha))
        if trainVmodel:
            if not hasattr(self, 'vModel'):
                print("Creating vModel")
                self.vModel = Sequential()
                for i in range(nb_layers):
                    self.vModel.add(Dense(layers_size[i], input_dim=state_size+1, activation='relu'))
                self.vModel.add(Dense(state_size, activation='linear'))
                self.vModel.compile(loss='mse', optimizer=Adam(lr=alpha))
        tot_reward_list = []
        for episode in range(total_episodes):
            tot_reward, nstep = GamePlayer.play_episode(self.env, train_function, self.memory)
            tot_reward_list.append(tot_reward)
            GamePlayer.model_fit_memory(state_size, nstep, gamma, reward_when_done, self.memory, self.model, self.vModel if trainVmodel else None)
            if logEvery > 0 and (episode+1) % logEvery == 0:
                print('Episode {} Average Reward: {}, alpha: {}'.format(episode+1, np.mean(tot_reward_list), K.eval(self.model.optimizer.lr)))
                tot_reward_list = []
    
    def dvn_action(self, state):
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        state = np.array(state).reshape(1, state_size)
        VValues = []
        for action in range(action_size):
            action = np.array(action).reshape(1, 1)
            vector = np.concatenate((state, action), axis=1)
            next_state = self.vModel.predict(vector)[0:state_size]
            VValues.append(max(self.model.predict(next_state)[0]))
        return random_argmax(VValues)

    def keras_dqn_dvn(self, N, total_episodes, layers_size=[32, 32], gamma=0.9, epsilon=0.2,
        decay_rate=0.0, alpha=0.001, reward_when_done=None, logEvery=None):
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        if logEvery is None:
            logEvery = N*2

        if logEvery > total_episodes:
            raise RangeError("logEvery should be lower than total_episodes")

        nb_layers = len(layers_size)
        if int(nb_layers) < 1:
            raise RangeError(nb_layers, len(layers_size))

        # Creating the model
        if not hasattr(self, 'model'):
            print("Creating model")
            self.model = Sequential()
            for i in range(nb_layers):
                self.model.add(Dense(layers_size[i], input_dim=state_size, activation='relu'))
            self.model.add(Dense(action_size, activation='linear'))
            self.model.compile(loss='mse', optimizer=Adam(lr=alpha))

        # Creating the vModel
        if not hasattr(self, 'vModel'):
            print("Creating vModel")
            self.vModel = Sequential()
            for i in range(nb_layers):
                self.vModel.add(Dense(layers_size[i], input_dim=state_size+1, activation='relu'))
            self.vModel.add(Dense(state_size, activation='linear'))
            self.vModel.compile(loss='mse', optimizer=Adam(lr=alpha))

        if not hasattr(self, 'memory'):
            self.memory = Memory(200000)

        tot_reward_list = []
        nbrecords = 0

        action_function = lambda state: GamePlayer.epsilon_action(state, self.env, epsilon, self.dvn_action)
        for episode in range(total_episodes):
            tot_reward, nstep = GamePlayer.play_episode(self.env, action_function, self.memory)
            nbrecords += nstep
            if nbrecords >= N:
                GamePlayer.model_fit_memory(state_size, N, gamma, reward_when_done, self.memory, self.model, self.vModel)
                nbrecords = 0
            tot_reward_list.append(tot_reward)

            if logEvery > 0 and (episode+1) % logEvery == 0:
                print('Episode {} Average Reward: {}, alpha: {}'.format(episode+1, np.mean(tot_reward_list), K.eval(self.model.optimizer.lr)))
                tot_reward_list = []
            if decay_rate != 0.0:
                epsilon = max(self.min_epsilon, epsilon*decay_rate)

        print("Total reward average:", np.mean(tot_reward_list))

    def keras_dqn_replay(self, N, total_episodes, layers_size=[24, 24], gamma=0.9, epsilon=0.2, 
        decay_rate=0.0, alpha=0.001, reward_when_done=None, logEvery=None):
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        if logEvery is None:
            logEvery = N*2

        if logEvery > total_episodes:
            raise RangeError("logEvery should be lower than total_episodes")

        nb_layers = len(layers_size)
        if int(nb_layers) < 1:
            raise RangeError(nb_layers, len(layers_size))

        # Creating the model
        if not hasattr(self, 'model'):
            print("Creating model")
            self.model = Sequential()
            for i in range(nb_layers):
                self.model.add(Dense(layers_size[i], input_dim=state_size, activation='relu'))
            self.model.add(Dense(action_size, activation='linear'))
            self.model.compile(loss='mse', optimizer=Adam(lr=alpha))

        if not hasattr(self, 'memory'):
            self.memory = Memory(200000)

        reward_list = []
        tot_reward_list = []
        nbrecords = 0

        action_function = lambda state: GamePlayer.epsilon_action(state, self.env, epsilon, self.keras_trained_action)
        for episode in range(total_episodes):
            tot_reward, nstep = GamePlayer.play_episode(self.env, action_function, self.memory)
            nbrecords += nstep
            if nbrecords >= N:
                GamePlayer.model_fit_memory(state_size, N, gamma, reward_when_done, self.memory, self.model)
                nbrecords = 0
            reward_list.append(tot_reward)

            if logEvery > 0 and (episode+1) % logEvery == 0:
                ave_reward = np.mean(reward_list)
                tot_reward_list.append(ave_reward)
                reward_list = []
                print('Episode {} Average Reward: {}, alpha: {}, e: {}'.format(episode+1, ave_reward, K.eval(self.model.optimizer.lr), epsilon))
            if decay_rate != 0.0:
                epsilon = max(self.min_epsilon, epsilon*decay_rate)

        print("Total reward average:", np.mean(tot_reward_list))

    def keras_dqn(self, N, total_episodes, layers_size=[24, 24], gamma=0.9, epsilon=0.2, 
        decay_rate=0.0, alpha=0.001, reward_when_done=None, logEvery=None):
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        if logEvery is None:
            logEvery = N*2

        if logEvery > total_episodes:
            raise RangeError("logEvery should be lower than total_episodes")

        nb_layers = len(layers_size)
        if int(nb_layers) < 1:
            raise RangeError(nb_layers, len(layers_size))

        # Creating the model
        if not hasattr(self, 'model'):
            self.model = Sequential()
            for i in range(nb_layers):
                self.model.add(Dense(layers_size[i], input_dim=state_size, activation='relu'))
            self.model.add(Dense(action_size, activation='linear'))
            self.model.compile(loss='mse', optimizer=Adam(lr=alpha))

        reward_list = []
        tot_reward_list = []
        nstep = 0
        Y = []
        S = []
        for episode in range(total_episodes):
            state = self.env.reset()
            done = False
            tot_reward = 0
            while done is False:
                if nstep == N:
                    S = np.stack(S, axis=0).reshape(N, state_size)
                    Y = np.stack(Y, axis=0)
                    self.model.fit(S, Y, epochs=1, verbose=0)
                    nstep = 0
                    Y = []
                    S = []

                state = np.array(state).reshape(1, state_size)
                S.append(state)
                Y.append(self.model.predict(state)[0])
                if np.random.rand(1) < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = random_argmax(Y[nstep])

                next_state, reward, done, _ = self.env.step(action)
                next_state = np.array(next_state).reshape(1, state_size)
                if done:
                    if reward_when_done is not None:
                        Y[nstep][action] = reward_when_done
                    else:
                        Y[nstep][action] = reward
                else:
                    Qnext = self.model.predict(next_state)[0]
                    Y[nstep][action] = reward + gamma * np.max(Qnext)

                state = next_state
                tot_reward += reward
                nstep += 1
            reward_list.append(tot_reward)

            if logEvery > 0 and (episode+1) % logEvery == 0:
                ave_reward = np.mean(reward_list)
                tot_reward_list.append(ave_reward)
                reward_list = []
                print('Episode {} Average Reward: {}, alpha: {}, e: {}'.format(episode+1, ave_reward, alpha, epsilon))
            if decay_rate != 0.0:
                epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-decay_rate*episode)

        print("Total reward average:", np.mean(tot_reward_list))

    def keras_trained_action(self, state):
        if self.model is None:
            raise ValueError("No model")
        state_size = self.env.observation_space.shape[0]
        return random_argmax(self.model.predict(np.array(state).reshape(1, state_size))[0])

    def tf_dqn(self, N, total_episodes, layers_size=[24, 24], gamma=0.9, epsilon=0.2, alpha=0.001, reward_when_done=None, logEvery=None):
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        if logEvery is None:
            logEvery = N*2

        if logEvery > total_episodes:
            raise RangeError("logEvery should be lower than total_episodes")

        nb_layers = len(layers_size)
        if int(nb_layers) < 1:
            raise RangeError(nb_layers, len(layers_size))

        # Creating the model
        inputs = tf.placeholder(shape=[1, state_size], dtype=tf.float32)
        training_inputs = tf.placeholder(shape=[N, state_size], dtype=tf.float32)
        states = tf.placeholder(shape=[N, state_size], dtype=tf.float32)
        ytarget = tf.placeholder(shape=[N, action_size], dtype=tf.float32)
        layers = [tf.Variable(tf.random_uniform([state_size, layers_size[0]], 0, 0.01))]
        predict_model = tf.nn.relu(tf.matmul(inputs, layers[0]))
        model = tf.nn.relu(tf.matmul(states, layers[0]))

        for i in range(1, nb_layers-2):
            layers.append(tf.Variable(tf.random_uniform([layers_size[i-1], layers_size[i]], 0, 0.01)))
            model = tf.nn.relu(tf.matmul(model, layers[i]))
            predict_model = tf.nn.relu(tf.matmul(predict_model, layers[i]))
        layers.append(tf.Variable(tf.random_uniform([layers_size[nb_layers-1], action_size], 0, 0.01)))
        model = tf.matmul(model, layers[nb_layers-1])
        predict_model = tf.matmul(predict_model, layers[nb_layers-1])

        loss = tf.reduce_sum(tf.square(ytarget - model))
        trainer = tf.train.AdamOptimizer(learning_rate=alpha)
        updateModel = trainer.minimize(loss)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            reward_list = []
            tot_reward_list = []
            nstep = 0
            Y = []
            S = []
            for episode in range(total_episodes):
                state = self.env.reset()
                done = False
                tot_reward = 0
                while done is False:
                    if nstep == N:
                        S = np.stack(S, axis=0).reshape(N, state_size)
                        Y = np.stack(Y, axis=0)
                        sess.run([updateModel], feed_dict={states: S, ytarget: Y})
                        nstep = 0
                        Y = []
                        S = []

                    state = np.array(state).reshape(1, state_size)
                    S.append(state)
                    Y.append(sess.run([predict_model], feed_dict={inputs: [state[0]]})[0][0])
                    if np.random.rand(1) < epsilon:
                        action = self.env.action_space.sample()
                    else:
                        action = random_argmax(Y[nstep])

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.array(next_state).reshape(1, state_size)
                    if done:
                        if reward_when_done is not None:
                            Y[nstep][action] = reward_when_done
                        else:
                            Y[nstep][action] = reward
                    else:
                        Qnext = sess.run([predict_model], feed_dict={inputs: [next_state[0]]})[0][0]
                        Y[nstep][action] = reward + gamma * np.max(Qnext)

                    state = next_state
                    tot_reward += reward
                    nstep += 1
                reward_list.append(tot_reward)

                if logEvery > 0 and (episode+1) % logEvery == 0:
                    ave_reward = np.mean(reward_list)
                    tot_reward_list.append(ave_reward)
                    reward_list = []
                    print('Episode {} Average Reward: {}, alpha: {}, e: {}'.format(episode+1, ave_reward, alpha, epsilon))

            self.model = predict_model
            self.input_placeholder = inputs
            print("Total reward average:", np.mean(tot_reward_list))

    def tf_trained_action(self, state):
        if self.model is None:
            raise ValueError("No model, consider training with tf_dqn function before calling this one")
        if self.input_placeholder is None:
            raise ValueError("No input_placeholder, consider training with tf_dqn function before calling this one")
        state_size = self.env.observation_space.shape[0]
        inputs = self.input_placeholder
        state = np.array(state).reshape(1, state_size)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            return random_argmax(sess.run([self.model], feed_dict={inputs: [state[0]]})[0][0])

def visualize_computer_playing(nb_episodes, env, action_function):
    with env:
        for episode in range(nb_episodes):
            state = env.reset()
            env.render()
            print("****************************************************")
            print("EPISODE ", episode)
            done = False
            tot_reward = 0
            while done is False:
                new_state, reward, done, info = env.step(action_function(state))
                env.render()
                state = new_state
                tot_reward += reward
            print("Reward:", tot_reward)
