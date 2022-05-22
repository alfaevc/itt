#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
import os 
import tqdm

import numpy as np
import es
import policy
from networks import NN

import tensorflow as tf

import matplotlib.pyplot as plt
import gym
import pybullet_envs

import time

time1=time.time()


"""### AT vs FD
"""
    

  
"""The cell below applies your ES implementation to the RL objective you've defined in the cell above."""
if __name__ == '__main__':
    # env_name = 'FetchPush-v1'
    # env_name = 'HalfCheetah-v2'
    env_name = 'Swimmer-v2'
    # env_name = 'InvertedPendulumBulletEnv-v0'

    # outfile = "files/twin_{}.txt".format(env_name)
    outfile = "files/twin_{}.txt".format(env_name+str(time.time()))
    with open(outfile, "w") as f:
        f.write("")

    env = gym.make(env_name)
    state_dim = env.reset().size
    print("The state dimension is " + str(state_dim))
    # theta_dim = state_dim + 1
    nA, = env.action_space.shape
    print("The action dimension is " + str(nA))

    actor_layers = [nA]
    critic_layers = [nA]

    # actor = NN(nA, layers=actor_layers)
    # actor.compile(optimizer=actor.optimizer, loss=actor.loss)
    # actor.fit(np.random.standard_normal((b,nA)), np.random.standard_normal((b,nA)), epochs=1, batch_size=b, verbose=0)

    # critic = NN(nA, layers=critic_layers)
    # critic.compile(optimizer=critic.optimizer, loss=actor.loss)
    # critic.fit(np.random.standard_normal((b,state_dim)), np.random.standard_normal((b,nA)), epochs=1, batch_size=b, verbose=0)

    # lp = policy.Log(env)
    #nn = NN(nA*2, layers=layers)
    #nn = NN(1, layers=layers)
    #nn.compile(optimizer=nn.optimizer, loss=nn.loss)
    #nn.fit(np.random.standard_normal((b,state_dim+nA)), np.random.standard_normal((b,1)), epochs=1, batch_size=b, verbose=0)
    #nn.fit(np.random.standard_normal((b,state_dim)), np.random.standard_normal((b,nA)), epochs=1, batch_size=b, verbose=0)
    #pi = policy.Energy(env, nn, state_dim, nA=nA)
    #pi = policy.GausNN(env, nn, state_dim, nA=nA)

    # theta_dim = nn.nnparams2theta().size

    # theta_dim=round((state_dim+nA)*(1+(state_dim+nA+1)/2))#num of polynomial terms up to degree 2
    # pi = policy.Energy_polyn(env, state_dim, nA=nA)
    # pi = policy.Energy_twin(env, actor, critic, state_dim, nA)
    # theta_dim = pi.actor_theta_len + pi.critic_theta_len


    num_seeds = 5
    max_epoch = 201
    res = np.zeros((num_seeds, max_epoch))
    method = "AT"
    print("The method is {}".format(method))

    for k in tqdm.tqdm(range(num_seeds)):
        
        
        actor = NN(nA, nA, layers=[nA])
        actor_theta = np.random.normal(size=(actor.theta_len,))
        actor(tf.keras.Input(shape=(nA,)))
        actor.update_params(actor.theta2nnparams(actor_theta, nA, nA))
        
        critic = NN(state_dim, nA, layers=[nA])
        critic_theta = np.random.normal(size=(critic.theta_len,))
        critic(tf.keras.Input(shape=(state_dim,)))
        critic.update_params(critic.theta2nnparams(critic_theta, state_dim, nA))
        
        
        # actor.print_params()

        pi = policy.Energy_twin(env, actor, critic, state_dim, nA)
        theta_dim = pi.actor_theta_len + pi.critic_theta_len
        

        # pi = policy.Gaus(env, env_name, state_dim, nA=nA)
        # theta_dim = (state_dim + 1) * 2 * nA

        print("Theta dimension is " + str(theta_dim))

        '''
        actor = NN(2*nA, layers=actor_layers)
        actor.compile(optimizer=actor.optimizer, loss=actor.loss)
        actor.fit(np.random.standard_normal((b,state_dim)), np.random.standard_normal((b,2*nA)), epochs=1, batch_size=b, verbose=0)
        pi = policy.GausNN(env, actor, state_dim, nA)
        theta_dim = pi.theta_len
        '''

        N = theta_dim

        #actor = NN(1, layers=layers)
        #actor = NN(nA*2, layers=layers)
        #actor.compile(optimizer=actor.optimizer, loss=actor.loss)
        #actor.fit(np.random.standard_normal((N,state_dim+nA)), np.random.standard_normal((N,1)), epochs=1, batch_size=N, verbose=0)
        #actor.fit(np.random.standard_normal((b,state_dim)), np.random.standard_normal((b,nA)), epochs=1, batch_size=b, verbose=0)
        # epsilons = np.random.multivariate_normal(np.zeros(5), np.identity(5), size=2)
        # print(epsilons)
        # fn = lambda x: fn_with_env(theta0 + x) * x
        # print(fn_with_env(theta0 + epsilons[0]) * epsilons[0])
        # print(np.array(list(map(fn, epsilons))))
        with open(outfile, "a") as f:
            f.write("Seed {}:\n".format(k))
        
        theta, accum_rewards = es.nn_twin_gradascent(actor, critic, pi, outfile, es.AT_gradient, es.rfF_arr, sigma=0.1, eta=1e-2, max_epoch=max_epoch, N=N)
        # theta, accum_rewards, method = es.gradascent_autoSwitch(theta0, pi, method=method, sigma=0.1, eta=1e-2, max_epoch=max_epoch, N=N)
        # theta0 = np.random.standard_normal(size=theta_dim)
        # theta, accum_rewards = es.gradascent(theta0, pi, outfile, es.AT_gradient_parallel, es.gaus_F_arr, sigma=1, eta=1e-2, max_epoch=max_epoch, N=N)
        # actor, accum_rewards = es.nn_gradascent(actor, pi, outfile, method=method, sigma=1, eta=1e-2, max_epoch=max_epoch, N=N)
        # nn_test_video(pi, actor, env_name, method)
        res[k] = np.array(accum_rewards)
    ns = range(1, len(accum_rewards)+1)

    time2=time.time()
    f.write("Time elapsed {}:\n".format(time2-time1))

    avs = np.mean(res, axis=0)
    maxs = np.max(res, axis=0)
    mins = np.min(res, axis=0)

    # method = "mixed"

    plt.fill_between(ns, mins, maxs, alpha=0.1)
    plt.plot(ns, avs, '-o', markersize=1, label=env_name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Iterations', fontsize = 15)
    plt.ylabel('Return', fontsize = 15)

    plt.title("Energy Twin {0} ES".format(method), fontsize = 24)
    plt.savefig("plots/Energy twin {0} ES {1}".format(method, env_name))






