import numpy as np
import argparse
# Assignment discription:
# In this problem, we will delve into the application of Markov Decision Processes (MDPs) 
# and the Value Iteration and Policy Iteration in the context of the Wumpus World. 
# The Wumpus World is a well-known problem in the field of artificial intelligence, 
# where the objective is to control an agent as it navigates through a grid-like environment 
# in search of gold, all while avoiding deadly pits and the formidable Wumpus creature.

# As you observed, the agent starts at the grid coordinate x = 0, y = 0, (x is the horizontal axes, y is the vertical axes) and its objectives are the following:
# - Finding the gold, which provides a significant positive reward (+10).
# - Avoiding the pits and the Wumpus, which are associated with negative penalties (-5 for each pit and -10 for the Wumpus).
# - Minimizing the incurred movement penalty (-0.4 for each non-goal cell). Due to the noise of the control signal, the movements are stochastic: There is an 80% chance that the agent moves in the intended direction. To be more specific, there is a 10% chance that the agent moves in one of the orthogonal directions. For example, if the agent intends to move UP, there’s an 80% chance it will move UP, a 10% chance it will move LEFT, and a 10% chance it will move RIGHT.

# There are three user-defined parameters in the program, namely, ‘gamma’, ‘eta’ and ‘max_iter’. The usages are the following:
# - gamma: sets a discount factor of future rewards. It represents how much future rewards are valued compared to immediate rewards.
# - eta: sets a threshold for the maximum value error between two adjacent iterations to assess algorithm convergence. If the maximum value error is less than this threshold, the iteration process is terminated.
# - max_iter: sets the maximum number of iterations that the implemented algorithm will run.



class WumpusWorld:
    def __init__(self):
        self.grid_size = 4 
        self.num_states = self.grid_size * self.grid_size 
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.rewards = self._initialize_rewards() 
        # print('rewards is:', self.rewards)
        self.transition_probabilities = self._initialize_transition_probabilities()
        # print('transition_probilities is:', self.transition_probabilities)

    def _initialize_rewards(self):
        rewards = np.full((self.grid_size, self.grid_size), -0.4)
        rewards[0, 3] = 10  # Gold position
        rewards[2, 2] = -5   # Pit position
        rewards[3, 2] = -5   # Pit position
        rewards[3, 1] = -10  # Wumpus position
        return rewards # return the rewards matrix

    def _initialize_transition_probabilities(self):
        # Transition probabilities for each action
        transition_probs = {} 
        for action in self.actions:
            transition_probs[action] = np.zeros((self.num_states, self.num_states))
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = i * self.grid_size + j
                for action in self.actions:
                    for next_action in self.actions:
                        if action == next_action:
                            prob = 0.8
                        elif (self._orthogonal(action, next_action)):
                            prob = 0.1
                        else:
                            continue
                        
                        next_state = self._move(i, j, next_action)
                        if next_state is not None:
                            transition_probs[action][state, next_state] = prob
        
        return transition_probs

    def _orthogonal(self, action1, action2): # check if two actions are orthogonal
        return (action1 in ['UP', 'DOWN'] and action2 in ['LEFT', 'RIGHT']) or \
               (action1 in ['LEFT', 'RIGHT'] and action2 in ['UP', 'DOWN'])

    def _move(self, i, j, action): # move to the next state
        if action == 'UP' and i > 0:
            return (i - 1) * self.grid_size + j
        elif action == 'DOWN' and i < self.grid_size - 1:
            return (i + 1) * self.grid_size + j
        elif action == 'LEFT' and j > 0:
            return i * self.grid_size + (j - 1)
        elif action == 'RIGHT' and j < self.grid_size - 1:
            return i * self.grid_size + (j + 1)
        return None

    def get_reward(self, state, action, next_state): # get the reward of the next state
        reward = self.rewards[next_state // self.grid_size, next_state % self.grid_size]
        return reward
    
    def get_transition_prob(self, state, action, next_state): # get the transition probability of the next state
        return self.transition_probabilities[action][state, next_state]

    def MDP_value_iteration(self, gamma, eta, max_iter): # value iteration algorithm to find the optimal value function
        V = np.zeros(self.num_states)
        # TODO, please use the value iteration algorithm mentioned in the lecture
        for i in range(max_iter): 
            V_new = np.zeros(self.num_states)
            for state in range(self.num_states): # for each state, calculate the value function
                action_values = []
                for action in self.actions: # for each action, calculate the value function
                    action_value = sum(
                        self.get_transition_prob(state, action, next_state) * 
                        (self.get_reward(state, action, next_state) + gamma * V[next_state])
                        for next_state in range(self.num_states) 
                    )
                    action_values.append(action_value)
                V_new[state] = max(action_values)
            if np.max(np.abs(V_new - V)) < eta:
                break
            V = V_new
        return V

    def MDP_policy_iteration(self, gamma, eta, max_iter): # policy iteration algorithm to find the optimal policy
        policy = np.random.choice(self.actions, size=self.num_states)
        V = np.zeros(self.num_states)
        # TODO, please use the policy iteration algorithm mentioned in the lecture 
        for i in range(max_iter):
            V = np.zeros(self.num_states)
            for _ in range(max_iter):
                V_new = np.zeros(self.num_states)
                for state in range(self.num_states):
                    action = policy[state]
                    V_new[state] = sum(
                        self.get_transition_prob(state, action, next_state) * 
                        (self.get_reward(state, action, next_state) + gamma * V[next_state])
                        for next_state in range(self.num_states)
                    )
                if np.max(np.abs(V_new - V)) < eta:
                    break
                V = V_new
            policy = self.MDP_policy(V, gamma)
        return V, policy
    
    def MDP_policy(self, V, gamma):
        # policy[s] is the best action to take in state s, firstly set it to 0 for all states
        policy = np.random.choice(self.actions, size=self.num_states)
        for state in range(self.num_states):
            action_values = []
            for action in self.actions:
                action_value = sum(
                    self.get_transition_prob(state, action, next_state) * 
                    (self.get_reward(state, action, next_state) + gamma * V[next_state])
                    for next_state in range(self.num_states)
                )
                action_values.append(action_value)
            policy[state] = self.actions[np.argmax(action_values)]
        return policy


def main(gamma, eta, max_iter):
    wumpus_world = WumpusWorld()
    
    # Value Iteration
    print(">>>>>>Running Value Iteration...")
    V_value = wumpus_world.MDP_value_iteration(gamma, eta, max_iter)
    print("Value Function (Value Iteration):\n", V_value.reshape(wumpus_world.grid_size, -1))
    print('Policy is:\n', wumpus_world.MDP_policy(V_value, gamma=gamma).reshape(wumpus_world.grid_size, -1))
    print(">>>>>-----------------------------")

    # Policy Iteration
    print(">>>>>>Running Policy Iteration...")
    V_policy, policy = wumpus_world.MDP_policy_iteration(gamma, eta, max_iter)
    print("Value Function (Policy Iteration):\n", V_policy.reshape(wumpus_world.grid_size, -1))
    print("Policy is:\n", policy.reshape(wumpus_world.grid_size, -1))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, required=True, help='Discount factor')
    parser.add_argument('--eta', type=float, required=True, help='Convergence threshold')
    parser.add_argument('--e', type=int, required=True, help='Maximum iterations')
    args = parser.parse_args()
    
    main(args.gamma, args.eta, args.e)