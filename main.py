import gym
import gym.spaces
import random
import numpy as np
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
#from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
import pickle

OU= OU()
#from collections import deque
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
import time
EPISODES = 2000
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001     #Target Network HyperParameters
LRA = 0.0001    #Learning rate for Actor
LRC = 0.001     #Lerning rate for Critic
EXPLORE = 100000.
epsilon = 1

def state_to_input(state):
    u,v=np.concatenate((state['observation'], state['desired_goal'])),\
    np.concatenate((state['observation'],state['achieved_goal']))
    #print(u.shape,v.shape)
    u,v = u.reshape((1,u.shape[0])),v.reshape((1,v.shape[0]))
    #print(u.shape,v.shape)
    return u,v


def ob_size(env):
    ob=env.reset()
    ob=state_to_input(ob)
    return ob[0].shape[1]


if __name__ == '__main__':


    env_name='FetchReach-v1'
    env = gym.make(env_name)
    action_dim = env.action_space.shape[0]
    state_dim = ob_size(env) #25 obs + 3 desired goal coordinates + 3 achieved goal coordinates
    save_path = "./save/"+env_name
    # Tensorflow config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)
    goals_buff = []

    try:
        import os
        os.makedirs(save_path)
    except:
        pass

    try:
        actor.model.load_weights(f"{save_path}/actormodel.h5")
        critic.model.load_weights(f"{save_path}/criticmodel.h5")
        actor.target_model.load_weights(f"{save_path}/actormodel.h5")
        critic.target_model.load_weights(f"{save_path}/criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")


    scores=np.array([])
    for i in range(EPISODES):
        state=env.reset()
        done=False
        total_reward = 0.
        episode_buff=[]
        loss = 0
        #Let the agent interact and walk through the episode
        while not done:

            epsilon -= 1.0 / EXPLORE
            noise_t = OU.function(x=np.zeros([1,action_dim]),mu=np.array([0.,0.5,-0.1,0.33]),\
            theta=np.array([0.6,1,1,0.6]), sigma=np.array([0.3,0.1,0.5,0.4]))

            s,s1 = state_to_input(state)


            act = actor.model.predict(s)

            ''' ADD noise to params
            '''
            act += noise_t


            next_state,reward,done,info=env.step(act[0])
            env.render()
            ns,ns1=state_to_input(next_state)
            achieved_goal = next_state['achieved_goal']
            substitute_goal = achieved_goal.copy()
            substitute_reward = env.compute_reward(achieved_goal, substitute_goal, info)

            buff.add(s, act[0], reward, ns, done)
            '''
            if done:
                goals_buff.append(next_state['achieved_goal'].copy())
                '''
            buff.add(s1,act[0],substitute_reward,ns1,True)
            total_reward += reward
            state = next_state
            # AGENT DONE INTERACTING

            #Do the batch update

            #sample the batch
            if buff.count()> BATCH_SIZE:
                batch = buff.getBatch(BATCH_SIZE)
                states = np.vstack([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.vstack([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                y_t = np.asarray([e[1] for e in batch])

                target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

                for k in range(BATCH_SIZE):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + GAMMA*target_q_values[k]

                loss += critic.model.train_on_batch([states,actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()


            '''

        # add replay data to the replay buffer
        #buff.batch_add(episode_buff)

        n_goals= 3
        if len(goals_buff) > n_goals:
            # set up HER
            #ADD hindsight
            # step 1. Sample random goals

            #HER_batch = [goals_buff[-1]]

            # TODO: Randomly sample goals
            HER_batch = random.sample(goals_buff,k=n_goals)

            for goal in HER_batch:
                #For each HER goal g', we calculate aux. reward
                # aux reward r = R(S,a,g')
                for state,act,_,next_state,done,info in episode_buff:
                    modified_state=state.copy()
                    modified_next_state = next_state.copy()

                    modified_state['desired_goal'] = \
                    modified_next_state['desired_goal'] = goal
                    aux_reward = env.compute_reward(modified_next_state['achieved_goal'], \
                    goal, info)


                #Append state||g', act, aux_r, next_state||g' to replay buffer
                    buff.add(modified_state, act, aux_reward, modified_next_state, done)


        '''





        print(f"Episode :{i+1} Reward :{total_reward} loss :{loss}")
        scores=np.append(scores,[total_reward, loss])

        if np.mod(i, 10) == 0:
            actor.model.save_weights(f"{save_path}/actormodel.h5", overwrite=True)
            with open(f"{save_path}/actormodel.json", "w") as outfile:
                json.dump(actor.model.to_json(), outfile)

            critic.model.save_weights(f"{save_path}/criticmodel.h5", overwrite=True)
            with open(f"{save_path}/criticmodel.json", "w") as outfile:
                json.dump(critic.model.to_json(), outfile)

            with open(f'{save_path}/metrics.dat','wb') as f:
                pickle.dump(scores,f)
