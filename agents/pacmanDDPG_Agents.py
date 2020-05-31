# Used code from
# DQN implementation by Tejas Kulkarni found at
# https://github.com/mrkulk/deepQN_tensorflow

# Used code from:
# The Pacman AI projects were developed at UC Berkeley found at
# http://ai.berkeley.edu/project_overview.html


import numpy as np
import random
import util
import time
import sys

# Pacman game
from pacman import Directions
from game import Agent
import game

# Replay memory
from collections import deque

# Neural nets
import tensorflow as tf
from statistics import mean

params = {
    # Model backups
    'load_file': None,
    'save_file': None,
    'save_interval' : 10000, 

    # Training parameters
    'train_start': 500,    # Episodes before training starts
    'batch_size': 32,       # Replay memory batch size
    'mem_size': 100000,     # Replay memory size

    'discount': 0.95,       # Discount rate (gamma value)
    'lr': .0002,            # Learning reate
    # 'rms_decay': 0.99,      # RMS Prop decay (switched to adam)
    # 'rms_eps': 1e-6,        # RMS Prop epsilon (switched to adam)

    # Epsilon value (epsilon-greedy)
    'eps': 0.5,             # Epsilon start value
    'eps_final': 0.1,       # Epsilon end value
    'eps_step': 10000       # Epsilon steps between start and end (linear)
}                     

# Actor that picks moves, takes in states and predicts best move to use based on gradients from Critic
class Actor(object):
    def __init__(self, params):
        self.params = params
        self.network_name = 'actor'
        self.sess = tf.Session()
        self.x = tf.placeholder('float', [None, params['width'],params['height'], 6],name=self.network_name + '_x')
        self.actions = tf.placeholder("float", [None, 4], name=self.network_name + '_actions')
        self.rewards = tf.placeholder("float", [None], name=self.network_name + '_rewards')
        self.terminals = tf.placeholder("float", [None], name=self.network_name + '_terminals')
        self.action_gradient = tf.placeholder(tf.float32, [1, 1, 4])

        layer_name = '1' ; size = 3 ; channels = 6 ; filters = 16 ; stride = 1
        self.w1 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
        self.c1 = tf.nn.conv2d(self.x, self.w1, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs')
        self.o1 = tf.nn.relu(tf.add(self.c1,self.b1),name=self.network_name + '_'+layer_name+'_activations')

        layer_name = '2' ; size = 3 ; channels = 16 ; filters = 32 ; stride = 1
        self.w2 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b2 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
        self.c2 = tf.nn.conv2d(self.o1, self.w2, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs')
        self.o2 = tf.nn.relu(tf.add(self.c2,self.b2),name=self.network_name + '_'+layer_name+'_activations')

        o2_shape = self.o2.get_shape().as_list()        
        layer_name = '3' ; hiddens = 256 ; dim = o2_shape[1]*o2_shape[2]*o2_shape[3]
        self.o2_flat = tf.reshape(self.o2, [-1,dim],name=self.network_name + '_'+layer_name+'_input_flat')
        self.w3 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b3 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
        self.ip3 = tf.add(tf.matmul(self.o2_flat,self.w3),self.b3,name=self.network_name + '_'+layer_name+'_ips')
        self.o3 = tf.nn.relu(self.ip3,name=self.network_name + '_'+layer_name+'_activations')

        layer_name = '4' ; hiddens = 4 ; dim = 256
        self.w4 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
        self.scaled_out = tf.add(tf.matmul(self.o3,self.w4),self.b4,name=self.network_name + '_'+layer_name+'_outputs')
  
        self.network_params = tf.trainable_variables()
        self.unnormalized_actor_gradients = tf.gradients(self.scaled_out, self.network_params)
        self.actor_gradients = list(map(lambda x: tf.div(x, params['batch_size']), self.unnormalized_actor_gradients))
 
        self.optim = tf.train.AdamOptimizer(self.params['lr']).apply_gradients(zip(self.actor_gradients, self.network_params))
        self.saver = tf.train.Saver(max_to_keep=0)

        self.sess.run(tf.global_variables_initializer())
        
    def train(self,bat_s,bat_a,bat_t,bat_n,bat_r,action_gradient):
        feed_dict={self.x: bat_s, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r, self.action_gradient: action_gradient}
        self.sess.run([self.optim],feed_dict=feed_dict)

    def save_ckpt(self,filename):
        self.saver.save(self.sess, filename)

# Critic that "judges" Actor, determines Q-score from memory produced by Actor moves and supplies gradients
class Critic(object):
    def __init__(self, params):
        self.params = params
        self.network_name = 'critic'
        self.sess = tf.Session()
        self.x = tf.placeholder('float', [None, params['width'],params['height'], 6],name=self.network_name + '_x')
        self.q_t = tf.placeholder('float', [None], name=self.network_name + '_q_t')
        self.actions = tf.placeholder("float", [None, 4], name=self.network_name + '_actions')
        self.rewards = tf.placeholder("float", [None], name=self.network_name + '_rewards')
        self.terminals = tf.placeholder("float", [None], name=self.network_name + '_terminals')
        self.action_grads = tf.placeholder("float",[1, 1, 4])

        # Layer 1 (Convolutional)
        layer_name = 'conv1' ; size = 3 ; channels = 6 ; filters = 16 ; stride = 1
        self.w1 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
        self.c1 = tf.nn.conv2d(self.x, self.w1, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs')
        self.o1 = tf.nn.relu(tf.add(self.c1,self.b1),name=self.network_name + '_'+layer_name+'_activations')

        # Layer 2 (Convolutional)
        layer_name = 'conv2' ; size = 3 ; channels = 16 ; filters = 32 ; stride = 1
        self.w2 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b2 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
        self.c2 = tf.nn.conv2d(self.o1, self.w2, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs')
        self.o2 = tf.nn.relu(tf.add(self.c2,self.b2),name=self.network_name + '_'+layer_name+'_activations')
        
        o2_shape = self.o2.get_shape().as_list()        

        # Layer 3 (Fully connected)
        layer_name = 'fc3' ; hiddens = 256 ; dim = o2_shape[1]*o2_shape[2]*o2_shape[3]
        self.o2_flat = tf.reshape(self.o2, [-1,dim],name=self.network_name + '_'+layer_name+'_input_flat')
        self.w3 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b3 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
        self.ip3 = tf.add(tf.matmul(self.o2_flat,self.w3),self.b3,name=self.network_name + '_'+layer_name+'_ips')
        self.o3 = tf.nn.relu(self.ip3,name=self.network_name + '_'+layer_name+'_activations')

        # Layer 4
        layer_name = 'fc4' ; hiddens = 4 ; dim = 256
        self.w4 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
        self.y = tf.add(tf.matmul(self.o3,self.w4),self.b4,name=self.network_name + '_'+layer_name+'_outputs')

        # Layer 4
        layer_name = 'fc5' ; hiddens = 4 ; dim = 4
        self.w4 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
        self.ip3 = tf.add(tf.matmul(self.actions,self.w4),self.b4,name=self.network_name + '_'+layer_name+'_outputs')
        self.o4 = tf.nn.relu(self.ip3,name=self.network_name + '_'+layer_name+'_activations')

        self.out = self.y + self.o4

        #Q,Cost,Optimizer
        self.discount = tf.constant(self.params['discount'])
        self.yj = tf.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.discount, self.q_t)))
        self.Q_pred = tf.reduce_sum(tf.multiply(self.y,self.actions), reduction_indices=1)
        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_pred), 2))
        
        #Q,Cost,Optimizer
        self.discount = tf.constant(self.params['discount'])
        self.yj = tf.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.discount, self.q_t)))
        self.Q_pred = tf.reduce_sum(tf.multiply(self.y,self.actions), reduction_indices=1)
        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_pred), 2))
        self.action_grads = tf.gradients(self.out, self.actions)

        if self.params['load_file'] is not None:
            self.global_step = tf.Variable(int(self.params['load_file'].split('_')[-1]),name='global_step', trainable=False)
        else:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        self.optim = tf.train.AdamOptimizer(self.params['lr']).minimize(self.cost, global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep=0)

        self.sess.run(tf.global_variables_initializer())

        if self.params['load_file'] is not None:
            print('Loading checkpoint...')
            self.saver.restore(self.sess,self.params['load_file'])

        
    def train(self,bat_s,bat_a,bat_t,bat_n,bat_r):
        feed_dict={self.x: bat_n, self.q_t: np.zeros(bat_n.shape[0]), self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        q_t = self.sess.run(self.y,feed_dict=feed_dict)
        q_t = np.amax(q_t, axis=1)
        feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        _,Q_value,cnt,cost = self.sess.run([self.optim, self.Q_pred, self.global_step,self.cost],feed_dict=feed_dict)
        return Q_value, cnt, cost

    def save_ckpt(self,filename):
        self.saver.save(self.sess, filename)

class PacmanDDPG(game.Agent):
    def __init__(self, args):

        # Load parameters from user-given arguments
        self.params = params
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['num_training'] = args['numTraining']

        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        self.actor = Actor(self.params)
        self.critic = Critic(self.params)

        # time started
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())

        self.Q_global = [-1]
        self.loss_global = [0]
        self.cost_disp = 0     

        # Stats
        self.cnt = self.critic.sess.run(self.critic.global_step)
        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()

    def getMove(self, state):
        # Exploit / Explore
        if np.random.rand() > params['eps']:
            # Exploit action
            # get scores for each move
            self.predicted_move_scores = self.actor.sess.run(
                self.actor.scaled_out,
                feed_dict = {self.actor.x: np.reshape(self.current_state,
                                                     (1, self.params['width'], self.params['height'], 6)), 
                             self.actor.actions: np.zeros((1, 4)),
                             self.actor.terminals: np.zeros(1),
                             self.actor.rewards: np.zeros(1)})[0]

            a_winner = np.argwhere(self.predicted_move_scores == np.amax(self.predicted_move_scores))

            # return best move
            if len(a_winner) > 1:
                move = self.get_direction(a_winner[np.random.randint(0, len(a_winner))][0])
            else:
                move = self.get_direction(a_winner[0][0])
        else:
            # Random:
            move = self.get_direction(np.random.randint(0, 4))

        # Save last_action
        self.last_action = self.get_value(move)

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
        if self.last_action is not None:
            # Process current experience state
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices(state)

            # Process current experience reward
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            if reward > 20:
                self.last_reward = 50.    # Eat ghost   (Yum! Yum!)
            elif reward > 0:
                self.last_reward = 10.    # Eat food    (Yum!)
            elif reward < -10:
                self.last_reward = -500.  # Get eaten   (Ouch!) -500
                self.won = False
            elif reward < 0:
                self.last_reward = -1.    # Punish time (Pff..)

            
            if(self.terminal and self.won):
                self.last_reward = 100.
            self.ep_rew += self.last_reward

            # Store last experience into memory 
            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Save model
            if(params['save_file']):
                if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
                    self.actor.save_ckpt('saves/model-' + params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
                    print('Model saved')

            # Train
            self.train()

        # Next
        self.local_cnt += 1
        self.frame += 1
        self.params['eps'] = max(self.params['eps_final'],
                                 1.00 - float(self.cnt)/ float(self.params['eps_step']))


    def observationFunction(self, state):
        # Do observation
        self.terminal = False
        self.observation_step(state)

        return state

    def final(self, state):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True
        self.observation_step(state)

        # Print stats
        log_file = open('./logs/'+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-'+str(self.params['height'])+'-x-'+str(self.params['num_training'])+'.log','a')
        log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        log_file.write("| Mean Critic Q-Value: %10f | won: %r \n" % ((mean(self.Q_global), self.won)))
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        sys.stdout.write("| Mean Critic Q-Value: %10f | won: %r \n" % ((mean(self.Q_global), self.won)))
        sys.stdout.flush()

    def train(self):
        # train after train_start incidents are in memory
        if (self.local_cnt > self.params['train_start']):
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = [] # States (s)
            batch_r = [] # Rewards (r)
            batch_a = [] # Actions (a)
            batch_n = [] # Next states (s')
            batch_t = [] # Terminal state (t)

            for i in batch:
                batch_s.append(i[0])
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3])
                batch_t.append(i[4])
            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = self.get_onehot(np.array(batch_a))
            batch_n = np.array(batch_n)
            batch_t = np.array(batch_t)

            # train critic on batch, get q value
            Q_value, self.cnt, self.cost_disp = self.critic.train(batch_s, batch_a, batch_t, batch_n, batch_r)
            self.Q_global.append(max(Q_value))
            self.loss_global.append(self.cost_disp)

            # get actor prediction on batch
            actor_prediction = self.actor.sess.run(
                self.actor.scaled_out,
                feed_dict = {self.actor.x: np.reshape(self.current_state,
                                                     (1, self.params['width'], self.params['height'], 6)), 
                             self.actor.actions: np.zeros((1, 4)),
                             self.actor.terminals: np.zeros(1),
                             self.actor.rewards: np.zeros(1)})

            # get sampled gradient from critic based on actor prediction
            action_grads = self.critic.sess.run(
                self.critic.action_grads,
                feed_dict = {self.critic.x: np.reshape(self.current_state,
                                                     (1, self.params['width'], self.params['height'], 6)), 
                             self.critic.actions: np.zeros((1, 4)),
                             self.critic.terminals: np.zeros(1),
                             self.critic.rewards: np.zeros(1)})

            # update actor policy
            self.actor.train(batch_s, batch_a, batch_t, batch_n, batch_r, action_grads)

    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total

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

    def registerInitialState(self, state): # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getStateMatrices(state)

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        # self.Q_global = [-1]
        # self.loss_global = [0]
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1

    def getAction(self, state):
        move = self.getMove(state)

        # Stop moving when not legal
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP

        return move
