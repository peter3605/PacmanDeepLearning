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
from statistics import mean

# Replay memory
from collections import deque

# Neural nets
import tensorflow as tf

params={
    # Model backups
    'load_file': None,
    'save_file': 'dddqn',
    'save_interval': 10000,

    # Training parameters
    'train_start': 500,    # Episodes before training starts
    'batch_size': 32,       # Replay memory batch size
    'mem_size': 100000,     # Replay memory size

    'discount': 0.95,       # Discount rate (gamma value)
    'lr': .0002,            # Learning reate
    # 'rms_decay': 0.99,      # RMS Prop decay (switched to adam)
    # 'rms_eps': 1e-6,        # RMS Prop epsilon (switched to adam)

    # Epsilon value (epsilon-greedy)
    'eps': 1.0,             # Epsilon start value
    'eps_final': 0.1,       # Epsilon end value
    'eps_step': 10000       # Epsilon steps between start and end (linear)
}

class DuelingDQN():
    def __init__(self, params, input_shape, num_actions, gamma=0.99):
        input_shape = (None,) + input_shape
        output_shape = (None, num_actions)
        self.sess = tf.Session()
        self.optimizer = tf.train.AdamOptimizer(params['lr'])
        self.gamma = 0.95
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.saver = tf.train.Saver(max_to_keep=0)

        self.qvars = None
        self.tvars = None

        self.input = tf.placeholder(tf.float32, shape=input_shape)

        self.params = params

        # q network graph definition
        with tf.variable_scope("qnet"):
            self.output = self._network('qnet', self.input, num_actions)
            current_scope = tf.get_default_graph().get_name_scope()
            self.qvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=current_scope)

        with tf.variable_scope("tnet"):
            self.target=self._network('tnet', self.input, num_actions)
            current_scope=tf.get_default_graph().get_name_scope()
            self.tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=current_scope)
            # self.tvars.sort(key=lambda x: x.name)

        # target network weights update operations
        self.update_target_op=[var[0].assign(var[1]) for var in zip(self.tvars, self.qvars)]

        # training operations definition
        self.yt_loss=tf.placeholder(tf.float32, shape=(None))
        self.actions=tf.placeholder(tf.float32, shape=(None))
        q_actions=tf.multiply(self.actions, self.output)

        self.loss=tf.losses.huber_loss(self.yt_loss, tf.reduce_sum(q_actions, axis=1))
        self.train_op=self.optimizer.minimize(loss=self.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())

        if self.params['load_file'] is not None:
            print('Loading checkpoint...')
            self.saver.restore(self.sess,self.params['load_file'])


    def _network(self, network_name, input, num_actions):
        self.network_name = network_name
        
        # Layer 1 (Convolutional)
        layer_name = 'conv1' ; size = 3 ; channels = 6 ; filters = 16 ; stride = 1
        self.w1 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
        self.c1 = tf.nn.conv2d(input, self.w1, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs')
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

        return self.y

    def _network2(self, network_name, input, num_actions):
        self.network_name = network_name

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

        #Q,Cost,Optimizer
        self.discount = tf.constant(self.params['discount'])
        self.yj = tf.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.discount, self.q_t)))
        self.Q_pred = tf.reduce_sum(tf.multiply(self.y,self.actions), reduction_indices=1)

        return self.Q_pred

    # update target network weights
    def updateTargetNetwork(self):
        self.sess.run(self.update_target_op)

    def updateModel(self, batch):
        states, actions, rewards, new_states, endgames = batch

        qtarget=None
        # computing target Q value using target network and double deep q-network algorithm
        [n_out, t_out]=self.sess.run([self.output, self.target], feed_dict={self.input: np.array(new_states)})
        target_action=np.argmax(n_out, axis=1)
        qtarget=np.array([output_sample[target_action[sample]] for sample, output_sample in enumerate(t_out)])
        yt = rewards + self.gamma * (np.logical_not(endgames) * qtarget)

        # computing loss and update weights of  Q network
        return self.sess.run([self.loss, self.train_op, self.global_step], feed_dict={self.input: np.array(states), self.yt_loss: yt, self.actions: np.array(actions)})

    def predict(self, X):
        return self.sess.run(self.output, feed_dict={self.input: X})

    def save_ckpt(self,filename):
        self.saver.save(self.sess, filename)


class PacmanDDDQN(game.Agent):
    def __init__(self, args):


        # Load parameters from user-given arguments
        self.params=params
        self.params['width']=args['width']
        self.params['height']=args['height']
        self.params['num_training']=args['numTraining']

        # Start Tensorflow session
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.qnet=DuelingDQN(
            self.params, (self.params['width'], self.params['height'], 6), 4)

        # time started
        self.general_record_time=time.strftime(
            "%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        # Q and cost
        self.Q_global=[-1]
        self.loss_global=[0]
        self.cost_disp=0

        # Stats
        self.cnt=self.qnet.sess.run(self.qnet.global_step)
        self.local_cnt=0

        self.numeps=0
        self.last_score=0
        self.s=time.time()
        self.last_reward=0.

        self.replay_mem=deque()
        self.last_scores=deque()


    def getMove(self, state):
        # Exploit / Explore
        if np.random.rand() > self.params['eps']:
            # Exploit action
            self.Q_pred=self.qnet.predict(np.reshape(self.current_state, (1, params['width'], params['height'], 6)))[0]

            self.Q_global.append(max(self.Q_pred))
            a_winner=np.argwhere(self.Q_pred == np.amax(self.Q_pred))

            if len(a_winner) > 1:
                move=self.get_direction(
                    a_winner[np.random.randint(0, len(a_winner))][0])
            else:
                move=self.get_direction(a_winner[0][0])
        else:
            # Random:
            move=self.get_direction(np.random.randint(0, 4))

        # Save last_action
        self.last_action=self.get_value(move)

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
            self.last_state=np.copy(self.current_state)
            self.current_state=self.getStateMatrices(state)

            # Process current experience reward
            self.current_score=state.getScore()
            reward=self.current_score - self.last_score
            self.last_score=self.current_score

            if reward > 20:
                self.last_reward=50.    # Eat ghost   (Yum! Yum!)
            elif reward > 0:
                self.last_reward=10.    # Eat food    (Yum!)
            elif reward < -10:
                self.last_reward=-500.  # Get eaten   (Ouch!) -500
                self.won=False
            elif reward < 0:
                self.last_reward=-1.    # Punish time (Pff..)


            if(self.terminal and self.won):
                self.last_reward=100.
            self.ep_rew += self.last_reward

            # Store last experience into memory
            experience=(self.last_state, float(self.last_reward),
                        self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Save model
            if(params['save_file']):
                if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
                    self.qnet.save_ckpt(
                        'saves/model-' + params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
                    print('Model saved')

            # Train
            self.train()

        # Next
        self.local_cnt += 1
        self.frame += 1
        self.params['eps']=max(self.params['eps_final'],
                                 1.00 - float(self.cnt) / float(self.params['eps_step']))


    def observationFunction(self, state):
        # Do observation
        self.terminal=False
        self.observation_step(state)

        return state

    def final(self, state):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal=True
        self.observation_step(state)

        # Print stats
        log_file=open('./logs/'+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-'+str(
            self.params['height'])+'-x-'+str(self.params['num_training'])+'.log', 'a')
        log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps, self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        log_file.write("| Main Q: %10f  | won: %r \n" %
                       (max(self.Q_global, default=np.nan), self.won))
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps, self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        sys.stdout.write("| Main Q: %10f | won: %r \n" %
                         (max(self.Q_global, default=np.nan), self.won))
        sys.stdout.flush()

    def train(self):
        # Train
        if (self.local_cnt > self.params['train_start']):
            batch=random.sample(self.replay_mem, self.params['batch_size'])
            batch_s=[]  # States (s)
            batch_r=[]  # Rewards (r)
            batch_a=[]  # Actions (a)
            batch_n=[]  # Next states (s')
            batch_t=[]  # Terminal state (t)

            for i in batch:
                batch_s.append(i[0])
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3])
                batch_t.append(i[4])
            batch_s=np.array(batch_s)
            batch_r=np.array(batch_r)
            batch_a=self.get_onehot(np.array(batch_a))
            batch_n=np.array(batch_n)
            batch_t=np.array(batch_t)

            self.cost_disp, _, self.cnt=self.qnet.updateModel((batch_s, batch_a, batch_r, batch_n, batch_t))


    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot=np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):
            actions_onehot[i][int(actions[i])]=1
        return actions_onehot

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices=np.swapaxes(stateMatrices, 0, 2)
        total=np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total

    def getStateMatrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height=state.data.layout.width, state.data.layout.height
            grid=state.data.layout.walls
            matrix=np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell=1 if grid[j][i] else 0
                    matrix[-1-i][j]=cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height=state.data.layout.width, state.data.layout.height
            matrix=np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos=agentState.configuration.getPosition()
                    cell=1
                    matrix[-1-int(pos[1])][int(pos[0])]=cell

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height=state.data.layout.width, state.data.layout.height
            matrix=np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos=agentState.configuration.getPosition()
                        cell=1
                        matrix[-1-int(pos[1])][int(pos[0])]=cell

            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height=state.data.layout.width, state.data.layout.height
            matrix=np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos=agentState.configuration.getPosition()
                        cell=1
                        matrix[-1-int(pos[1])][int(pos[0])]=cell

            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height=state.data.layout.width, state.data.layout.height
            grid=state.data.food
            matrix=np.zeros((height, width), dtype=np.int8)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell=1 if grid[j][i] else 0
                    matrix[-1-i][j]=cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height=state.data.layout.width, state.data.layout.height
            capsules=state.data.layout.capsules
            matrix=np.zeros((height, width), dtype=np.int8)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]]=1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height
        width, height=self.params['width'], self.params['height']
        observation=np.zeros((6, height, width))

        observation[0]=getWallMatrix(state)
        observation[1]=getPacmanMatrix(state)
        observation[2]=getGhostMatrix(state)
        observation[3]=getScaredGhostMatrix(state)
        observation[4]=getFoodMatrix(state)
        observation[5]=getCapsulesMatrix(state)

        observation=np.swapaxes(observation, 0, 2)

        return observation

    def registerInitialState(self, state):  # inspects the starting state

        # Reset reward
        self.last_score=0
        self.current_score=0
        self.last_reward=0.
        self.ep_rew=0

        # Reset state
        self.last_state=None
        self.current_state=self.getStateMatrices(state)

        # Reset actions
        self.last_action=None

        # Reset vars
        self.terminal=None
        self.won=True
        self.Q_global=[]
        self.delay=0

        # Next
        self.frame=0
        self.numeps += 1

    def getAction(self, state):
        move=self.getMove(state)

        # Stop moving when not legal
        legal=state.getLegalActions(0)
        if move not in legal:
            move=Directions.STOP

        return move
