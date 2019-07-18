import random
from math import log10, floor
import numpy as np

# Keras DQN
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Tensorflow DQN
import tensorflow as tf

def round_to(x, sig_figs):
    if x == 0.0:
        return 0.0
    if x < 0.0:
        return round(x, -int(floor(log10(abs(-x))) - (sig_figs - 1)))
    return round(x, -int(floor(log10(abs(x))) - (sig_figs - 1)))

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
    
    def epison_q_action(self, state, epsilon):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(self.Q(state, self.qtable))
        # Else doing a random choice --> exploration
        else:
            action = self.env.action_space.sample()
        return action

    def epison_double_q_action(self, state, epsilon, Q1, Q2):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            Q = [x + y for x, y in zip(self.Q(state, Q1), self.Q(state, Q2))]
            action = np.argmax(Q)
        # Else doing a random choice --> exploration
        else:
            action = self.env.action_space.sample()
        return action
    
    def start_game(self, render = False):
        state = self.env.reset()
        if (render):
            self.env.render()
        return state

    def q_trained_action(self, state):
        return np.argmax(self.Q(state, self.qtable))

    def double_trained_action(self, state):
        Q = [x + y for x, y in zip(self.Q(state, self.qtable), self.Q(state, self.Q2))]
        return np.argmax(Q)
        
    def play_game_step(self, action, render = True):
        new_state, reward, done, info = self.env.step(action)
        if (render):
            self.env.render()
        return new_state, reward, done, info

    def close(self):
        self.end_game()

    def end_game(self):
        self.env.close() 
        
    def train(self, total_episodes, alpha, gamma, epsilon, decay_rate, logEvery = 100):
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

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
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


    def double_q_train(self, total_episodes, alpha, gamma, epsilon, decay_rate, logEvery = 100):
        self.start_game(False)
        # Exploration parameters
        alpha0 = alpha
        Q1 = {}
        Q2 = {}
        
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
                    print(self.Q(state,Q1))
                    print(self.Q(state,Q2))
                    raise IndexError
                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, done, info = self.play_game_step(action, False)

                tradeoff = random.uniform(0, 1)
                if tradeoff > 0.5:
                    self.Q(state, Q1)[action] += alpha * (reward  + gamma * self.Q(new_state, Q2)[np.argmax(self.Q(state, Q1))] - self.Q(state, Q1)[action])
                else:
                    self.Q(state, Q2)[action] += alpha * (reward  + gamma * self.Q(new_state, Q1)[np.argmax(self.Q(state, Q2))] - self.Q(state, Q2)[action])

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

    def adversarial_q_train(self, total_episodes, alpha, gamma, epsilon, decay_rate, logEvery = 100):
        self.start_game(False)
        # Exploration parameters
        alpha0 = alpha
        Q1 = {}
        Q2 = {}
        Q = (Q1, Q2)
        
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
            tot_reward_1 = 0
            tot_reward_2 = 0
            tot_reward = [tot_reward_1, tot_reward_2]
            nstep = 0
            while done is False:
                action = self.epison_double_q_action(state, epsilon, Q1, Q2)
                if action >= self.env.action_space.n:
                    raise IndexError
                try:
                    # Take the action (a) and observe the outcome state(s') and reward (r)
                    new_state, reward, done, info = self.env.play_symbol(action, (nstep%2)+1)
                    new_state = self.state_function(new_state)
                except:
                    self.Q(state, Q1)[action] = -2
                    self.Q(state, Q2)[action] = -2
                    tot_reward[(nstep)%2] -= 2
                    continue
                
                if done:
                    if previous_state[0][previous_action] != 0 or state[0][action] != 0:
                        raise IndexError("weird", done, action, state, previous_state, previous_action)
                    #print("Adversary winning reward:", reward, new_state, state, previous_state, previous_action, action)
                    self.Q(previous_state, Q[(nstep+1)%2])[previous_action] = -reward
                    self.Q(state, Q[(nstep)%2])[action] = reward
                    tot_reward[(nstep+1)%2] += -reward
                    tot_reward[(nstep)%2] += reward
                else:
                    self.Q(previous_state, Q[(nstep+1)%2])[previous_action] += alpha * (reward  - gamma * max(self.Q(new_state, Q[(nstep)%2])) - self.Q(state, Q[(nstep+1)%2])[action])
                    self.Q(state, Q[(nstep)%2])[action] += alpha * (reward  - gamma * max(self.Q(new_state, Q[(nstep+1)%2])) - self.Q(state, Q[(nstep)%2])[action])
                    tot_reward[(nstep)%2] += reward

                if new_state == state or previous_state == new_state:
                    raise IndexError("What is going on ?", new_state, state, previous_state, action, done)

                nstep += 1

                # Our new state is state
                previous_state = state
                state = new_state
                previous_action = action
            reward_list.append(tot_reward)
            if decay_rate != 0.0:
                epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-decay_rate*episode)
            
            if logEvery > 0 and (episode+1) % logEvery == 0:
                ave_reward = np.mean(reward_list)
                tot_reward_list.append(ave_reward)
                reward_list = []
                #alpha = alpha0 * (total_episodes - episode)/total_episodes
                print('Episode {} Average Reward: {}, alpha: {}, e: {}, len(Q) {}'.format(episode+1, ave_reward, alpha, epsilon, len(self.qtable)))
        self.qtable = Q1
        self.Q2 = Q2
        return tot_reward_list

    def keras_dqn(self, N, total_episodes, layers_size=[24, 24], gamma=0.9, epsilon=0.2, alpha=0.001, reward_when_done=None, logEvery=None):
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
        model = Sequential()
        for i in range(nb_layers):
            model.add(Dense(layers_size[i], input_dim=state_size, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=alpha))

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
                    model.fit(S, Y, epochs=1, verbose=0)
                    nstep = 0
                    Y = []
                    S = []

                state = np.array(state).reshape(1, state_size)
                S.append(state)
                Y.append(model.predict(state)[0])
                if np.random.rand(1) < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(Y[nstep])

                next_state, reward, done, _ = self.env.step(action)
                next_state = np.array(next_state).reshape(1, state_size)
                if done:
                    if reward_when_done is not None:
                        Y[nstep][action] = reward_when_done
                    else:
                        Y[nstep][action] = reward
                else:
                    Qnext = model.predict(next_state)[0]
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

        self.model = model
        print("Total reward average:", np.mean(tot_reward_list))

    def keras_trained_action(self, state):
        if self.model is None:
            raise ValueError("No model")
        state_size = self.env.observation_space.shape[0]
        return np.argmax(self.model.predict(np.array(state).reshape(1, state_size))[0])

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
        inputs = tf.placeholder(shape=[1,state_size],dtype=tf.float32)
        training_inputs = tf.placeholder(shape=[N,state_size],dtype=tf.float32)
        states = tf.placeholder(shape=[N,state_size],dtype=tf.float32)
        ytarget = tf.placeholder(shape=[N,action_size],dtype=tf.float32)
        layers = [tf.Variable(tf.random_uniform([state_size, layers_size[0]],0,0.01))]
        predict_model = tf.nn.relu(tf.matmul(inputs, layers[0]))
        model = tf.nn.relu(tf.matmul(states, layers[0]))

        for i in range(1, nb_layers-2):
            layers.append(tf.Variable(tf.random_uniform([layers_size[i-1], layers_size[i]],0,0.01)))
            model = tf.nn.relu(tf.matmul(model, layers[i]))
            predict_model = tf.nn.relu(tf.matmul(predict_model, layers[i]))
        layers.append(tf.Variable(tf.random_uniform([layers_size[nb_layers-1], action_size],0,0.01)))
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
                        action = np.argmax(Y[nstep])

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
            return np.argmax(sess.run([self.model], feed_dict={inputs: [state[0]]})[0][0])

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
                state = new_state
                tot_reward += reward
            print("Reward:", tot_reward)