import numpy as np
import gym
import random
from collections import defaultdict

class Agent:
    
    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
        self.epsilon = 1    # Exploration rate
        self.decay = 0.5
        self.gamma = 0.99    # Discount rate
        self.alpha = 0.5
        
        self.learning_method = "EXPECTED_SARSA" # implemented methods are SARSA, SARSA_MAX, EXPECTED_SARSA
        self.i_episode = 0
        
        self.epsilon_function = self.decayed_epsilon
        def calculate(next_state):
            return self.Q[next_state][self.select_action(next_state)]
        self.next_value_function = calculate
        

    def decayed_epsilon(self):
        self.epsilon *= self.decay
            
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        
        probs = np.ones(self.nA) * self.epsilon /self.nA
        probs[np.argmax(self.Q[state])] += 1 - self.epsilon

        return np.random.choice(np.arange(self.nA), p=probs)
        
        

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        if (done == False):
            self.epsilon = 1.0 / (1.0 + self.i_episode)
            if(self.learning_method == "SARSA"):
                next_action = self.select_action(state)
                self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
            elif(self.learning_method == "SARSA_MAX"):
                next_action = self.select_action(state)
                self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
            elif(self.learning_method == "EXPECTED_SARSA"):
                probs = np.ones(self.nA) * self.epsilon /self.nA
                probs[np.argmax(self.Q[state])] += 1 - self.epsilon
                next_action = np.random.choice(np.arange(self.nA), p=probs)
                self.Q[state][action] += self.alpha * (reward + self.gamma * np.sum(self.Q[next_state] * probs)  - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
            self.i_episode +=  1