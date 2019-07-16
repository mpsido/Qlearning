import random
from math import log10, floor
import numpy as np

def round_to(x, sig_figs):
    if x == 0.0:
        return 0.0
    if x < 0.0:
        return round(x, -int(floor(log10(abs(-x))) - (sig_figs - 1)))
    return round(x, -int(floor(log10(abs(x))) - (sig_figs - 1)))

class GamePlayer:

    def __init__(self, env, state_function):
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

    def computer_play_step(self, state):
        action = np.argmax(self.Q(state, self.qtable))
        return self.play_game_step(action)

    def double_trained_computer_play_step(self, state):
        Q = [x + y for x, y in zip(self.Q(state, self.qtable), self.Q(state, self.Q2))]
        action = np.argmax(Q)
        return self.play_game_step(action)
        
    def play_game_step(self, action, render = True):
        new_state, reward, done, info = self.env.step(action)
        if (render):
            self.env.render()
        return new_state, reward, done, info

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

    def adversarial_q_train(self, total_episodes, alpha, gamma, epsilon, decay_rate, adversary, logEvery = 100):
        self.start_game(False)
        # Exploration parameters
        alpha0 = alpha
        
        reward_list = []
        tot_reward_list = []
        # 2 For life or until learning is stopped
        for episode in range(total_episodes):
            # Reset the environment
            state = self.env.reset()
            state = self.state_function(state)
            new_state = ()
            previous_state = ()
            previous_action = -1
            done = False
            tot_reward = 0
            while done is False:
                if (state[1] == -1 and self.symbol == 1) or (state[1] != -1 and state[1] != self.symbol):
                    action = self.epison_q_action(state, epsilon)
                    if action >= self.env.action_space.n:
                        raise IndexError
                    try:
                        # Take the action (a) and observe the outcome state(s') and reward (r)
                        new_state, reward, done, info = self.env.play_symbol(action, self.symbol)
                        new_state = self.state_function(new_state)
                    except:
                        self.Q(state, self.qtable)[action] = -2
                        tot_reward -= 2
                        continue
                else:
                    try:
                        action = adversary.epison_q_action(state, 0.5)
                        new_state, reward, done, info = self.env.play_symbol(action, adversary.symbol)
                        new_state = self.state_function(new_state)
                        if done:
                            if previous_state[0][previous_action] != 0:
                                raise IndexError("weird", done, action, state, previous_state)
                            #print("Adversary winning reward:", reward, new_state, state, previous_state, previous_action, action)
                            self.Q(previous_state, self.qtable)[previous_action] = -reward
                            tot_reward += -reward
                            break
                    except:
                        continue

                # dirty ugly cheat witchcraft that does not even work
                # reward += abs(new_state[0])+10.0*abs(new_state[1])

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                if not done:
                    self.Q(state, self.qtable)[action] += alpha * (reward  + gamma * max(self.Q(new_state, self.qtable)) - self.Q(state, self.qtable)[action])
                else:
                    if state[0][action] != 0:
                        raise IndexError("weird", done, action, new_state, state, previous_state)
                    self.Q(state, self.qtable)[action] = reward

                if new_state == state or previous_state == new_state:
                    raise IndexError("What is going on ?", new_state, state, previous_state, action, done)

                # Our new state is state
                previous_state = state
                state = new_state
                previous_action = action
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

def visualize_computer_playing(nb_episodes, trained_game, isDoubleTrained = False):
    for episode in range(nb_episodes):
        state = trained_game.start_game(True)
        print("****************************************************")
        print("EPISODE ", episode)
        done = False
        tot_reward = 0
        while done is False:
            if isDoubleTrained:
                new_state, reward, done, info = trained_game.double_trained_computer_play_step(state)
            else:
                new_state, reward, done, info = trained_game.computer_play_step(state)
            state = new_state
            tot_reward += reward
        print("Reward:", tot_reward)
    trained_game.end_game()