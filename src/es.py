import numpy as np

from networks import NN

import gym
import pybullet_envs
import time

import math
import multiprocessing as mp
from itertools import repeat
import tensorflow as tf
from itertools import product

"""NN interface
"""
def update_nn_params(input_nn,new_params):
    params = list(input_nn.parameters())
    current_index= 0
    for i in range(len(params)):
        shape = params[i].data.detach().numpy().shape
        if len(shape)>1:#params[i] is 2d tensor
            arr = new_params[current_index:current_index+shape[0]*shape[1]].reshape(shape)
            params[i].data = (torch.from_numpy(arr)).float()
            current_index+=shape[0]*shape[1]
        else:#params[i] is 1d tensor
            arr = new_params[current_index:current_index+shape[0]]
            params[i].data = (torch.from_numpy(arr)).float()
            current_index+=shape[0]

def get_nn_dim(input_nn):
    params = list(input_nn.parameters())
    counter = 0
    for i in range(len(params)):
        shape = params[i].data.detach().numpy().shape
        if len(shape)>1:#params[i] is 2d tensor
            counter+=shape[0]*shape[1]
        else:#params[i] is 1d tensor
            counter+=shape[0]
    return counter

class state_tower(nn.Module):
    def __init__(self):
        super(state_tower, self).__init__()
        state_dim = env.reset().size
        nA, = env.action_space.shape
        self.fc1 = nn.Linear(state_dim, nA, bias=False)  
        self.fc2 = nn.Linear(nA, nA, bias=False)
        self.fc3 = nn.Linear(nA, nA, bias=False)

def state_feed_forward(state_net,state):#have to separate feed_forward from the class instance, otherwise multiprocessing raises errors
    x = (torch.from_numpy(state)).float()
    #x = torchF.relu(state_net.fc1(x))
    x = state_net.fc1(x)
    x = torchF.relu(x)
    x = state_net.fc2(x)
    x = torchF.relu(x)
    x = state_net.fc3(x)
    latent_state = x.detach().numpy()
    #latent_state = latent_state/sum(np.abs(latent_state)) #normalize
    return latent_state

class action_tower(nn.Module):
    def __init__(self):
        super(action_tower, self).__init__()
        nA, = env.action_space.shape
        self.fc1 = nn.Linear(nA, nA, bias=False)#can automate this. create nn for any given input layer dimensions, instead of fixed dimensions  
        self.fc2 = nn.Linear(nA, nA, bias=False)
        

def action_feed_forward(action_net,action):#have to separate feed_forward from the class instance, otherwise multiprocessing raises errors
    x = (torch.from_numpy(action)).float()
    #x = torchF.relu(action_net.fc1(x))
    x = action_net.fc1(x)#can automate this. feedforward given nn dimensions
    x = torchF.relu(x)
    x = action_net.fc2(x)
    latent_action = x.detach().numpy()
    return latent_action

def get_state_net(theta):
    action_net = action_tower()
    action_nn_dim = get_nn_dim(action_net)
    state_net = state_tower()
    update_nn_params(state_net, theta[action_nn_dim:])
    return state_net

def get_action_net(theta):
    action_net = action_tower()
    action_nn_dim = get_nn_dim(action_net)
    update_nn_params(action_net,theta[:action_nn_dim])
    return action_net
    
def get_latent_actions(actions_arr,theta):
    action_net = get_action_net(theta)
    return action_feed_forward(action_net,actions_arr)

def get_theta_dim():
    state_net = state_tower()
    action_net = action_tower()
    state_nn_dim = get_nn_dim(state_net)
    action_nn_dim = get_nn_dim(action_net)
    return action_nn_dim+state_nn_dim

"""fitness functions for parallel
"""
def get_output(output):
    nA=int(round(output.shape[1]/2))
    min_logvar=1
    max_logvar=3
    means = output[:, 0:nA]
    raw_vs = output[:, nA:]
    logvars = max_logvar - tf.nn.softplus(max_logvar - raw_vs)
    logvars = min_logvar + tf.nn.softplus(logvars - min_logvar)
    return means, tf.exp(logvars).numpy()

def gaus_F(theta, env_name, gamma=1, max_step=5e3):
    env = gym.make(env_name)#this takes no time
    nA, = env.action_space.shape
    G = 0.0
    done = False
    discount = 1
    steps = 0
    state = env.reset()
    a_dim = np.arange(nA)
    state_dim = state.size
    steps_count=0#cannot use global var here because subprocesses do not have access to global var
    # while not done:
    while not done and (steps < max_step):
        # WRITE CODE HERE
        fn = lambda a: [theta[2*a*(state_dim+1)] + state @ theta[2*a*(state_dim+1)+1: (2*a+1)*(state_dim+1)], 
                        theta[(2*a+1)*(state_dim+1)] + state @ theta[(2*a+1)*(state_dim+1)+1: (2*a+2)*(state_dim+1)]]
        mvs = np.array(list(map(fn, a_dim))).flatten()
        a_mean, a_v  = get_output(np.expand_dims(mvs, 0))
        action = np.tanh(np.random.normal(a_mean[0], a_v[0]))
        # action = np.random.normal(a_mean[0], a_v[0])

        state, reward, done, _ = env.step(action)
        steps_count+=1
        G += reward * discount
        discount *= gamma
        steps += 1
    return G,steps_count

def gaus_F_arr(epsilons, env_name, sigma, theta):
    grad = np.zeros(epsilons.shape)
    steps_count = 0
    for i in range(epsilons.shape[0]):#for loop inefficient. to be improved
        output1, time1 = gaus_F(theta + sigma * epsilons[i], env_name)
        output2, time2 = gaus_F(theta - sigma * epsilons[i], env_name)
        grad[i] = (output1 - output2) * epsilons[i]
        steps_count += time1+time2
    grad = np.average(grad,axis=0)/sigma/2
    return [grad,steps_count]
        
    #fn = lambda x: (F(theta + sigma * x) - F(theta - sigma * x)) * x
    #return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma/2

def gaus_eval(theta, env_name):
    env = gym.make(env_name)#this takes no time
    nA, = env.action_space.shape
    G = 0.0
    done = False
    steps = 0
    state = env.reset()
    a_dim = np.arange(nA)
    state_dim = state.size
    while not done:
        # WRITE CODE HERE
        fn = lambda a: [theta[2*a*(state_dim+1)] + state @ theta[2*a*(state_dim+1)+1: (2*a+1)*(state_dim+1)], 
                        theta[(2*a+1)*(state_dim+1)] + state @ theta[(2*a+1)*(state_dim+1)+1: (2*a+2)*(state_dim+1)]]
        mvs = np.array(list(map(fn, a_dim))).flatten()
        a_mean, a_v  = get_output(np.expand_dims(mvs, 0))
        action = np.tanh(np.random.normal(a_mean[0], a_v[0]))
        # action = np.random.normal(a_mean[0], a_v[0])

        state, reward, done, _ = env.step(action)
        G += reward
        steps += 1
    return G

def energy_actions(actor, critic, state, nA, K=10):
    sample_actions = np.array(list(product([-1,0,1], repeat=nA*K)))
    # sample_actions = np.random.uniform(low=-1.0, high=1.0, size=(K,nA))
    latent_actions, latent_states = actor(sample_actions).numpy(), np.tile(critic(np.expand_dims(state,0)).numpy().reshape(-1), (K,1))
    energies = np.einsum('ij,ij->i', latent_actions, latent_states)
    return sample_actions[np.argmin(energies)]
     
def energy_min_action(actor, critic, state):
    param1 = actor.get_layer_i_param(0)
    param2 = actor.get_layer_i_param(1)
    latent_state = critic(np.expand_dims(state,0)).numpy()
    return np.dot(np.dot(param1, param2), latent_state.T)

def twin_F(theta, env_name, gamma=1, max_step=1e4):
    env = gym.make(env_name)
    nA, = env.action_space.shape
    state = env.reset()
    state_dim = state.size

    actor = NN(nA, nA, layers=[nA])
    actor_theta = np.random.normal(size=(actor.theta_len,))
    actor(tf.keras.Input(shape=(nA,)))
    actor.update_params(actor.theta2nnparams(actor_theta, nA, nA))
    
    critic = NN(state_dim, nA, layers=[nA])
    critic_theta = np.random.normal(size=(critic.theta_len,))
    critic(tf.keras.Input(shape=(state_dim,)))
    critic.update_params(critic.theta2nnparams(critic_theta, state_dim, nA))

    actor_theta_len = actor.nnparams2theta().size

    steps_count=0
    
    G = 0.0
    done = False
    discount = 1
    actor.update_params(actor.theta2nnparams(theta[:actor_theta_len], nA, nA))
    critic.update_params(critic.theta2nnparams(theta[actor_theta_len:], state_dim, nA))
    while not done:
        action = energy_actions(actor, critic, state, nA, K=nA*10)
        # action = energy_min_action(actor, critic, state)
        state, reward, done, _ = env.step(action)
        G += reward * discount
        discount *= gamma
        steps_count+=1
    return G, steps_count

def twin_F_arr(epsilons, sigma, theta, env_name):
    grad = np.zeros(epsilons.shape)
    steps_count = 0
    for i in range(epsilons.shape[0]):
        #can be made more efficient. but would not improve runtime, since only loop <=8 times
        output1, time1 = twin_F(theta + sigma * epsilons[i], env_name)
        output2, time2 = twin_F(theta - sigma * epsilons[i], env_name)
        grad[i] = (output1 - output2) * epsilons[i]
        steps_count += time1+time2
    grad = np.average(grad,axis=0)/sigma/2
    #fn = lambda x: (F(theta + sigma * x) - F(theta - sigma * x)) * x
    #return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma/2
    return [grad,steps_count]

def rf_energy_action(nA, table, latent_state, all_actions):
    max_depth = 2*nA
    left,right = 0,len(table)-1#left end and right end of search region. we iteratively refine the search region
    currentDepth = 0
    while currentDepth < max_depth:
        #go to next level of depth
        mid = (left+right)/2#not an integer
        left_latent_action_sum = table[math.floor(mid)] - table[left]
        left_prob = np.exp(left_latent_action_sum@latent_state) #make cause overflow or underflow. need some normalization
        
        right_latent_action_sum = table[right] - table[math.ceiling(mid)]
        right_prob = np.exp(right_latent_action_sum@latent_state)
        
        p = left_prob/(left_prob+right_prob)
        coin_toss = np.random.binomial(1, p)
        if coin_toss == 1:#go left
            right=math.floor(mid)
        else:#go right
            left = math.ceiling(mid)
        currentDepth+=1
    return all_actions[left]

def get_latent_action(action, theta):
    return action

def get_latent_state(state, theta):
    return state

def rfF(theta, env_name, gamma=1, max_step=5e3):
    gym.logger.set_level(40)
    env = gym.make(env_name)#this takes no time
    nA, = env.action_space.shape
    G = 0.0
    done = False
    discount = 1
    state = env.reset()
    a_dim = np.arange(nA)
    state_dim = state.size
    steps_count=0#cannot use global var here because subprocesses do not have access to global var
    #preprocessing
    all_actions = np.array([i for i in product([-1,-1/3, 0, 1/3,1],repeat=nA)])#need to make the number of actions some power of 2
    
    fn = lambda a: get_latent_action(a, theta)
    table = np.cumsum(np.array(list(map(fn, all_actions))), axis=0)

    while not done:
        latent_state = get_latent_state(state, theta)
        action = rf_energy_action(nA, table, latent_state, all_actions)
        state, reward, done, _ = env.step(action)
        steps_count+=1
        G += reward * discount
        discount *= gamma
    return G, steps_count

def rfF_arr(epsilons, sigma, theta):
    grad = np.zeros(epsilons.shape)
    steps_count = 0
    for i in range(epsilons.shape[0]):
        #can be made more efficient. but would not improve runtime, since only loop <=8 times
        output1, time1 = rfF(theta + sigma * epsilons[i])
        output2, time2 = rfF(theta - sigma * epsilons[i])
        grad[i] = (output1 - output2) * epsilons[i]
        steps_count += time1+time2
    grad = np.average(grad,axis=0)/sigma/2
    return [grad,steps_count]

def twin_eval(theta, env_name):
    env = gym.make(env_name)
    nA, = env.action_space.shape
    state = env.reset()
    state_dim = state.size

    b=1
    actor = NN(nA, layers=[2*nA])
    actor.compile(optimizer=actor.optimizer, loss=actor.loss)
    actor.fit(np.random.standard_normal((b,nA)), np.random.standard_normal((b,nA)), epochs=1, batch_size=b, verbose=0)
    critic = NN(nA, layers=[nA])
    critic.compile(optimizer=critic.optimizer, loss=actor.loss)
    critic.fit(np.random.standard_normal((b,state_dim)), np.random.standard_normal((b,nA)), epochs=1, batch_size=b, verbose=0)
    actor_theta_len = actor.nnparams2theta().size

    G = 0.0
    done = False
    actor.update_params(actor.theta2nnparams(theta[:actor_theta_len], nA, nA))
    critic.update_params(critic.theta2nnparams(theta[actor_theta_len:], state_dim, nA))
    
    all_actions = np.array([i for i in product([-1,-1/3,1/3,1],repeat=nA)])#need to make the number of actions some power of 2
    
    fn = lambda a: get_latent_action(a, theta)
    table = np.cumsum(np.array(list(map(fn, all_actions))), axis=0)
    
    while not done:
        # action = energy_actions(actor, critic, state, nA, K=nA*10)
        # action = energy_min_action(actor, critic, state)
        latent_state = get_latent_state(state, theta)
        action = rf_energy_action(nA, table, latent_state, all_actions)
        state, reward, done, _ = env.step(action)
        G += reward
    return G

"""### AT vs FD
"""

def orthogonal_epsilons(N,dim):
    #assume input N is a multiple of dim. 
    epsilons_N=np.zeros((N,dim))    
    for i in range(0,round(N/dim)):
      epsilons = np.random.standard_normal(size=(dim, dim))
      Q, _ = np.linalg.qr(epsilons)#orthogonalize epsilons
      Q_normalize=np.copy(Q)
      fn = lambda x, y: np.linalg.norm(x) * y
      #renormalize rows of Q by multiplying it by length of corresponding row of epsilons
      Q_normalize = np.array(list(map(fn, epsilons, Q_normalize)))
      epsilons_N[i*dim:(i+1)*dim] = Q_normalize@Q
    #for i in range(theta.size):
    #  norm=np.linalg.norm(epsilons[i])
    #  Q_normalize[i]=Q_normalize[i]*norm
    return epsilons_N

def hessian_gaussian_smoothing(theta, policy, sigma=1, N=100):
    epsilons = orthogonal_epsilons(N,theta.size)
    fn = lambda x: (np.outer(x,x)- np.identity(theta.size))*policy.F(theta + sigma * x)/(sigma**2)
    hessian = np.mean(np.array(list(map(fn, epsilons))), axis=0) 
    return hessian

def vanilla_gradient(theta, policy, sigma=1, N=100):
    epsilons=orthogonal_epsilons(N,theta.size)
    fn = lambda x: policy.F(theta + sigma * x) * x
    return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma

def FD_gradient(theta, policy, sigma=1, N=100):
    # epsilons = np.random.standard_normal(size=(N, theta.size))
    epsilons=orthogonal_epsilons(N,theta.size)
    G = policy.F(theta)
    fn = lambda x: (policy.F(theta + sigma * x) - G) * x
    return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma

def AT_gradient(theta, policy, env_name, sigma=1, N=100):
    #epsilons = np.random.standard_normal(size=(N, theta.size))
    epsilons=orthogonal_epsilons(N,theta.size)
    fn = lambda x: (policy.F(theta + sigma * x) - policy.F(theta - sigma * x)) * x
    return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma/2

def collect_result(result):
    result_list.append(result[0])
    steps_list.append(result[1]) 

def AT_gradient_parallel(theta, F, env_name, sigma=1, N=100):
    numCPU=mp.cpu_count()
    pool=mp.Pool(numCPU)
    jobs=math.ceil(N/numCPU)
    if math.floor(N/numCPU) >= 0.8*N/numCPU:
        jobs = math.floor(N/numCPU)
    # N=jobs*numCPU

    epsilons=orthogonal_epsilons(N,theta.size)
    global result_list#must be global for callback function to edit
    result_list = []
    global steps_list
    steps_list = []
    for i in range(numCPU):
        pool.apply_async(F, args = (epsilons[i*jobs:(i+1)*jobs], env_name, sigma, theta),callback=collect_result)
    pool.close()
    pool.join()
    #print('AT grad: ',np.mean(result_list))
    # result_list = F_arr(epsilons,sigma,theta)
    #results=pool.starmap(F_arr,zip(list(repeat(epsilons[0],numCPU)),list(repeat(sigma,numCPU)),list(repeat(theta,numCPU))))
    return np.mean(result_list)    


def choose_covariate(theta, policy, sigma=1, N=100):
    grad=AT_gradient(theta, policy, sigma=sigma, N=2*N)
    hessian=hessian_gaussian_smoothing(theta, policy, sigma=sigma, N=N)
    MSE_AT=(np.linalg.norm(grad)**2)/N
    MSE_FD=np.copy(MSE_AT)
    MSE_FD+=((N+4)*sigma**4/(4*N))*np.linalg.norm(hessian, ord='fro')**2
    diag_hess = np.diagonal(hessian)
    MSE_FD+=(2.5*sigma**4/N) * diag_hess @ diag_hess
    choice = "AT" if (2*N/(N+1))*MSE_AT > MSE_FD else "FD"
    return choice, MSE_FD, MSE_AT

'''
def gradascent_autoSwitch(theta0, policy, method=None, sigma=0.1, eta=1e-2, max_epoch=200, N=100):
    theta = np.copy(theta0)
    accum_rewards = np.zeros(max_epoch)
    for i in range(max_epoch):
      accum_rewards[i] = policy.eval(theta)
      print("The return for episode {0} is {1}".format(i, accum_rewards[i]))
      if i%10==0:#update method every 20 iterations
        choice, MSE_FD, MSE_AT = choose_covariate(theta,policy,sigma,N=theta.size*5)
        method=choice
        print("method updated to: ", method,', MSE of FD is ', MSE_FD,', MSE OF AT is ', MSE_AT)    
      
      if method == "AT":
        theta += eta * AT_gradient(theta, policy, sigma, N=N)
      else:
        theta += eta * FD_gradient(theta, policy, sigma, N=2*N)#make # of queries for FD and AT the same   

    return theta, accum_rewards, method
'''



def gradascent(theta0, policy, filename, grad, F, sigma=1, eta=1e-3, max_epoch=200, N=100):
    theta = np.copy(theta0)
    accum_rewards = np.zeros(max_epoch)
    for i in range(max_epoch): 
      accum_rewards[i] = policy.eval(theta)
      if i%1==0:
        print("The return for epoch {0} is {1}".format(i, accum_rewards[i]))    
        with open(filename, "a") as f:
          f.write("%.d %.2f \n" % (i, accum_rewards[i]))

      theta += eta * grad(theta, F, policy.env_name, sigma, N=N)
      # theta += eta * grad(theta, policy, policy.env_name, sigma, N=N)

    return theta, accum_rewards

def nn_gradascent(actor, policy, filename, grad, F, sigma=1, eta=1e-3, max_epoch=200, N=100):
    accum_rewards = np.zeros(max_epoch)
    theta = actor.nnparams2theta()
    # actor.print_params()
    for i in range(max_epoch):  
      theta += eta * grad(theta, F, sigma, N=N)
      if i%1==0:
        new_params = actor.theta2nnparams(theta, policy.input_dim, policy.nn.output_dim)
        actor.update_params(new_params)
        accum_rewards[i] = policy.eval(actor)
        print("The return for epoch {0} is {1}".format(i, accum_rewards[i]))
        with open(filename, "a") as f:
          f.write("%.d %.2f \n" % (i, accum_rewards[i]))

    return actor, accum_rewards

def nn_twin_gradascent(actor, critic, policy, filename, grad, F, sigma=1, eta=1e-3, max_epoch=200, N=100):
    accum_rewards = np.zeros(max_epoch)

    # print(actor.nnparams2theta().size)
    # actor.print_params()
    # print(critic.nnparams2theta().size)
    # critic.print_params()
    theta = np.concatenate((actor.nnparams2theta(), critic.nnparams2theta()))
    # print(theta.size)
    for i in range(max_epoch):
      theta += eta * grad(theta, policy, sigma, N=N)
      # theta += eta * grad(theta, F, sigma, N=N)
      if i%1==0:
        theta_action = theta[:policy.actor_theta_len]
        theta_state = theta[policy.actor_theta_len:]
        act_params = actor.theta2nnparams(theta_action, policy.nA, policy.nA)
        actor.update_params(act_params)
        critic_params = critic.theta2nnparams(theta_state, policy.state_dim, policy.nA)
        critic.update_params(critic_params)
        accum_rewards[i] = policy.eval(actor, critic)
        print("The return for epoch {0} is {1}".format(i, accum_rewards[i]))
        with open(filename, "a") as f:
          f.write("%.d %.2f \n" % (i, accum_rewards[i]))

    return actor, accum_rewards

    



