from pacman import Directions
from game import Agent
import game

from DDQN import DDQN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import sys
import shelve
from collections import deque


#model_trained = False
mode = 'test' # options: 'train', 'test', 'resume'

discount_factor = 0.95  # discount factor
learning_rate = 0.0002  # learning rate
tau = 0.07  # tau used in when updating target model: formula on line 108

rms_alpha = 0.99    # alpha value for RMSprop optimizer
rms_epsilon = 1e-06 # epsilon used in optimizer for numerical stability

batch_size = 32     # memory replay batch size
memory_size = 10000     # memory replay limit
start_training = 1000  # minimum memory size before training begins
update_target_iter = 50  # update target network params after these many iterations

epsilon = 1.0       # starting epsilon of model
epsilon_final = 0.11   # epsilon will not drop below this during training
epsilon_step = 3000   # rate epsilon decays (higher = slower decay): formula on line 168

save_iterations = 50 # how often to save the model (by episode): more often = slower
write_iterations = 10 # how often to record the results of the model (rewars, wins, qvalues, epsilon)


    
class PacmanDDQN:
    def __init__(self, args):

        self.mode = mode

         # Width and height arguments to set the size of the state matrices
        self.width = args['width']
        self.height = args['height']
        self.num_training = args['numTraining']

        # Utilize gpu is possible, otherwise use cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_mem = shelve.open('ddqn_data/replay.pickle', writeback=True)
		
        # Mode will determine which files we open, how we write to them, and the epsilon value
        if self.mode == 'train':
            print('MODEL HAS BEGUN TRAINING')
            self.epsilon = epsilon
            self.replay_mem['experiences'] = deque(maxlen=memory_size)
            self.q_model = DDQN(width=self.width, height=self.height).double().to(self.device)
            self.target_model = DDQN(self.width, self.height).double().to(self.device)
            self.updateTargetModel(initialize=True)
            self.rewards_file = open('ddqn_data/ddqn_train_rewards.txt', 'w')
            self.wins_file = open('ddqn_data/ddqn_train_wins.txt', 'w')
            self.qvalue_file = open('ddqn_data/ddqn_train_qvalues.txt', 'w')
            self.variables_file = open('ddqn_data/ddqn_train_variables.txt', 'w')
        elif self.mode == 'resume':
            print('MODEL IS CONTINUING TRAINING')
            self.variables = open('ddqn_data/ddqn_train_variables.txt', 'r').readline().split(',')[-5:-1]
            self.q_model = torch.load('ddqn_data/model.pt').to(self.device)
            self.target_model = torch.load('ddqn_data/target.pt').to(self.device)
            self.rewards_file = open('ddqn_data/ddqn_train_rewards.txt', 'a')
            self.wins_file = open('ddqn_data/ddqn_train_wins.txt', 'a')
            self.qvalue_file = open('ddqn_data/ddqn_train_qvalues.txt', 'a')
            self.variables_file = open('ddqn_data/ddqn_train_variables.txt', 'a')
        else:
            print('MODEL HAS BEGUN TESTING')
            self.epsilon = 0.0
            self.q_model = torch.load('ddqn_data/model.pt').to(self.device)
            self.target_model = torch.load('ddqn_data/target.pt').to(self.device)
            self.rewards_file = open('ddqn_data/ddqn_test_rewards.txt', 'w')
            self.wins_file = open('ddqn_data/ddqn_test_wins.txt', 'w')
            self.qvalue_file = open('ddqn_data/ddqn_test_qvalues.txt', 'w')

        # Using RMSprop optimizer
        self.optim = torch.optim.RMSprop(self.q_model.parameters(), lr=learning_rate, alpha=rms_alpha, eps=rms_epsilon)

        # Use L1Smooth for our loss function
        self.loss_function = torch.nn.SmoothL1Loss()
        
        # If resuming training we need to grab old values
        if self.mode != 'resume':
            # Track total wins and all frames
            self.episode_number = 0
            self.win_counter = 0
            self.total_frames = 0
        else:
            self.episode_number = int(self.variables[0])
            self.epsilon = float(self.variables[1])
            self.win_counter = int(self.variables[2])
            self.total_frames = int(self.variables[3])
        
        # Rewards/scores for pacman
        self.last_score = 0.
        self.last_reward = 0.

        # Track Q values from each step/episode
        self.Q_global = [] 

        self.save_iter = save_iterations
        self.write_iter = write_iterations
        
		 
    # Intiialize: copy all weights over, otherwise: copy weights with tau discount formula
    def updateTargetModel(self, initialize):
        if initialize:
            self.target_model.load_state_dict(self.q_model.state_dict())
        else:
            for target_param, param in zip(self.target_model.parameters(), self.q_model.parameters()):
                target_param.data.copy_(tau * param + (1 - tau) * target_param)

    
    # Called from pacman.py, resets terminal state and has observationStep do all the work
    def observationFunction(self, state):
        # do observation
        self.terminal = False
        self.observationStep(state)

        return state 


    # Runs before executing action to compute the reward/training based on last action taken
    def observationStep(self, state):
        # If this is not the first move we need to check how the last action did
        if self.last_action is not None:
            self.last_state = self.current_state
            self.current_state = self.getStateMatrices(state)

            self.current_score = state.getScore() # get the current score displayed on screen
            reward = self.current_score - self.last_score # check how the reward has changed since last action
            self.last_score = self.current_score
            
            # Assign awards based on what pacman did from the last state and the current state
            if reward > 20:
                self.last_reward = 50.    # pacman ate a ghost
            elif reward > 0:
                self.last_reward = 10.    # pacman ate a food pellet
            elif reward < -10:
                self.last_reward = -500.  # pacman was killed by a ghost
                self.won = False
            elif reward < 0:
                self.last_reward = -1.    # pacman did not eat anything

            # This is only triggered if the terminal state resulted in pacman winning the game
            if self.terminal and self.won:
                self.last_reward = 100.
                self.win_counter += 1

            self.episode_reward += self.last_reward

            # Experience tuple that is stored in our replay buffer
            experience = (self.last_state, self.last_reward, self.last_action, self.current_state, self.terminal)
            self.replay_mem['experiences'].append(experience)

            # If model is set to training mode we want to compute/update q values
            if self.mode != 'test':
                self.train()

        # Track frames passed
        self.total_frames += 1
        self.episode_frames += 1
		
		# If training, decay the epsilon, but do not go below the minimum epsilon specified
        if self.mode != 'test':
            self.epsilon = max(epsilon_final, 1.00 - (float(self.episode_number) / float(epsilon_step)))


    # Bulk of algorthim which updates networks based on Q values 
    def train(self):
        # Only train after collecting the minimum # of experiences
        if self.total_frames > start_training:
            # Extract experience tuple from replay buffer
            batch = random.sample(self.replay_mem['experiences'], batch_size)
            states, rewards, actions, next_states, terminals = zip(*batch)
            
            # Convert all to tensors, states need to be stacked because stored as tuples
            states = torch.from_numpy(np.stack(states)).to(self.device)
            rewards = torch.DoubleTensor(rewards).unsqueeze(1).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            next_states = torch.from_numpy(np.stack(next_states)).to(self.device)
            terminals = torch.ByteTensor(terminals).unsqueeze(1).to(self.device)
            
            # Q model qvalues for current states
            curr_model_Q = self.q_model(states).gather(1, actions)

            # Q model values and Target model values for next states
            next_model_Q = self.q_model(next_states)
            next_target_V = self.target_model(next_states)

            # Extract max column of Q values from Q model
            next_model_Q = next_model_Q.detach().max(1)[0]
            next_model_Q = next_model_Q.unsqueeze(1)
            
            # Extract max column of V values from target model
            next_target_V = next_target_V.detach().max(1)[0]
            next_target_V = next_target_V.unsqueeze(1)
    
            # Extract the minmium for each experience between our two models
            min_network_value = torch.min(next_model_Q, next_target_V)

            # Formula to compute the expected q value for the DDQN model
            expected_Q = rewards + (1 - terminals) * discount_factor * min_network_value
            
			# Compute the loss between observed q value and expected q value
            loss = self.loss_function(curr_model_Q, expected_Q)
            
			# optimize model - update weights
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    # Called from pacman.py, uses getMove to get the actual action to be taken
    def getAction(self, state):
        move = self.getMove(state)

        # If the action chosen is illegal then pacman just stays put
        if move not in state.getLegalActions(0):
            move = Directions.STOP
        return move


    # Finds the best action based on Q value or Epsilon (randomness)
    def getMove(self, state):
         # Exploration vs Exploitation
        if np.random.rand()  > self.epsilon: # Exploit the optimal action by qvalue
            #current_state = torch.from_numpy(np.stack(self.current_state))
            current_state = torch.from_numpy(self.current_state)
            current_state = current_state.unsqueeze(0) # add a channel for state to go through CNN
            current_state = current_state.to(self.device)

			# Get qvalues for each action from the model
            qvalues = self.q_model(current_state)
            qvalues = qvalues.view(-1)
            best_q = qvalues.max().item()
            best_action = (best_q == qvalues).nonzero().flatten()
            
            self.Q_global.append(best_q) # keep track of this actions qvalue
			
            # If two actions tied for highest qvalue then we choose one randomly
            if len(best_action) > 1:
                best_action_value = best_action[torch.randint(0, len(best_action), (1,))].item()
                action = self.get_direction(best_action_value)
            else:
                action = self.get_direction(best_action.item())
        else: # Explore a random action instead of exploit
            best_action_value = torch.randint(0, 4, (1,)).item()
            action = self.get_direction(best_action_value)

        # Store last action for following iteration
        self.last_action = self.get_value(action)

        return action


    # Last function called per episode to do evaluation and writing/saving
    def final(self, state):
        self.episode_reward += self.last_reward
        self.terminal = True
        self.observationStep(state) # Check final state for win or lose

        # Write information about training to train/test files
        if self.episode_number % self.write_iter == 0:
            self.rewards_file.write(str(self.episode_reward) + ',')
            self.wins_file.write(str((self.win_counter/self.episode_number)*100.0) + ',')
            self.qvalue_file.write(str(max(self.Q_global, default=float('nan'))) + ',')

        # Save the model every N iterations, specified by user
        if self.episode_number % self.save_iter == 0 and self.mode != 'test':
            torch.save(self.q_model, 'ddqn_data/model.pt')
            torch.save(self.target_model, 'ddqn_data/target.pt')
            self.variables_file.write(str(self.episode_number) + ',' + str(self.epsilon) + ',' +\
                                      str(self.win_counter) + ',' + str(self.total_frames) + ',')

        # Target network is updated every N iterations, specified by user
        if self.episode_number % update_target_iter == 0 and mode != 'test':
            self.updateTargetModel(initialize=False)

        # Written by episode: episode number, # actions, best q value, reward, epsilon, win rate
        sys.stdout.write('# %4d | A: %5d | W: %r ' % (self.episode_number, self.episode_frames, self.won))
        sys.stdout.write('| R: %12f | E: %10f ' % (self.episode_reward, self.epsilon))
        sys.stdout.write('| Q: %10f ' % (max(self.Q_global, default=float('nan'))))
        sys.stdout.write('| WR: %4f\n' % (float(self.win_counter/self.episode_number)*100.0))
        sys.stdout.flush()

        # Pickle the replay memory if resumining training
        if self.episode_number == self.num_training:
            self.replay_mem.close()
	        

    # Initial state of an episode to set up all values we will use
    def registerInitialState(self, state):
        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.episode_reward = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getStateMatrices(state)

        # Reset actions
        self.last_action = None

        # Reset state variables
        self.terminal = None
        self.won = True
        self.Q_global = []

        # Track each frame (actions taken) and episode
        self.episode_frames = 0
        self.episode_number += 1


    # Get numerical value for the action taken
    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        if direction == Directions.EAST:
            return 1.
        if direction == Directions.SOUTH:
            return 2.
        if direction == Directions.WEST:
            return 3.


    # Inverse to get_value function
    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        if value == 1.:
            return Directions.EAST
        if value == 2.:
            return Directions.SOUTH
        if value == 3.:
            return Directions.WEST


    # Convert the game state into a matrix for the CNN
    def getStateMatrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """ 
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height 
        width, height = self.width, self.height
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)
        return observation

	
                    

                    
