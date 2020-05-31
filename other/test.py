from network import *
import numpy as np
import torch
import random
import util
import time
import sys

import game
from game import Agent
from pacman import Directions

from collections import deque



class PacmanTD3(Agent):

    def __init(self, args, max_action=None):

        print('Initializing TD3 Agent')

        self.train_start = 5000
        self.batch_size = 32
        self.mem_size = 100000
        self.discount = 0.95
        self.learning_rate = 0.0002

        self.epsilon = 1.0
        self.ending_epsilon = 0.1

        self.actor = Actor(12, 256, 4, None)
        self.actor_target = Actor(12, 256, 4, None)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.critic = Critic(12, 256, 4, None)
        self.critic_target = Critic(12, 256, 4, None)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        self.max_action = max_action

        self.local_count = 0

        self.replay_mem = deque()
        self.last_scores = deque()


    def getMove(self, state):
        print('We are Stopping')
        return 'Stop'


    def getAction(self, state):
        move = self.getMove(state)

        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP

        return move


    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        elif direction == Directions.EAST:
            return 1.
        elif direction == Directions.SOUTH:
            return 2.
        else:
            return 3.


    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        elif value == 1.:
            return Directions.EAST
        elif value == 2.:
            return Directions.SOUTH
        else:
            return Directions.WEST


    def observation_step(self, state):
        next_action = 


    def observationFunction(self, state):
        # Do observation
        self.terminal = False
        self.observation_step(state)

        return state


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
        width, height = self.params['width'], self.params['height']
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)

        observation = np.swapaxes(observation, 0, 2)

        return observation

    

    def train(self):
        if self.local_count > self.train_start:
            batch = random.sample(self.replay_mem, self.batch_size)
            batch_states = []
            batch_rewards = []
            batch_actions = []
            batch_next_states = []
            batch_terminals = []

            for i in batch:
                batch_states.append(i[0])
                batch_rewards.append(i[1])
                batch_actions.append(i[2])
                batch_next_states.append(i[3])
                batch_terminals.append(i[4])
            batch_states = torch.FloatTensor(batch_states)
            batch_rewards = torch.FloatTensor(batch_rewards)
            batch_actions = torch.FloatTensor(batch_actions)
            batch_next_states = torch.FloatTensor(batch_next_states)
            batch_terminals = torch.FloatTensor(batch_terminals)



