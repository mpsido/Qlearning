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
        
    def erase_training(self):
        self.qtable = {}
    
    def Q(self, state):
        s = self.state_function(state)
        if s not in self.qtable:
            self.qtable[s] = []
            for i in range(self.env.action_space.n):
                self.qtable[s].append(0.0)
        return self.qtable[s]
    
    def epison_q_action(self, state, epsilon):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(self.Q(state))
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
        action = np.argmax(self.Q(state))
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
        
        has_won = False
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
                self.Q(state)[action] += alpha * (reward  + gamma * max(self.Q(new_state)) - self.Q(state)[action])

                # Our new state is state
                state = new_state
                tot_reward += reward
            reward_list.append(tot_reward)
            if decay_rate != 0.0:
                epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-decay_rate*episode)
            
            if logEvery >0 and (episode+1) % logEvery == 0:
                ave_reward = np.mean(reward_list)
                tot_reward_list.append(ave_reward)
                reward_list = []
                #alpha = alpha0 * (total_episodes - episode)/total_episodes
                print('Episode {} Average Reward: {}, alpha: {}, e: {}, len(Q) {}'.format(episode+1, ave_reward, alpha, epsilon, len(self.qtable)))
        return tot_reward_list