import numpy as np
import gym
import network
import utils
import pybullet_envs

def energy_action_hash(latent_actions_arr, actions_arr, latent_state, medians, intervals, projection_vector,powers):
    binary_rep = (np.sign(projection_vector @ latent_state-medians) + 1) / 2
    query = int(round(binary_rep @ powers))
    latent_actions_queried = latent_actions_arr[intervals[query,0]:intervals[query,1]]
    ind = np.argmax(latent_actions_queried @ latent_state)
    return actions_arr[intervals[query,0]+ind]

def energy_action_iot(iot_net,state,hyper_params):
    n_batch = hyper_params.n_batch; batch_size = hyper_params.batch_size; nA = hyper_params.nA
    energies=np.zeros(n_batch*batch_size)
    actions_arr = np.random.uniform(-1,1,size=(n_batch*batch_size,nA))
    state_arr = np.tile(state,(n_batch*batch_size,1))
    state_action_arr = np.concatenate((state_arr, actions_arr), axis=1)
    for i in range(n_batch):
        energies[i*batch_size:(i+1)*batch_size] = network.iot_forward(iot_net,state_action_arr[i*batch_size:(i+1)*batch_size])[:,0]
    return actions_arr[np.argmin(energies)]


def energy_action_itt(action_net, latent_state,hyper_params):
    n_batch = hyper_params.n_batch; batch_size = hyper_params.batch_size; nA = hyper_params.nA
    actions_arr = np.random.uniform(-1,1,size=(n_batch*batch_size,nA))
    latent_actions_arr = np.zeros((n_batch * batch_size, nA))
    for i in range(n_batch):
        latent_actions_arr[batch_size * i:batch_size * (i + 1)] = network.itt_action_forward(
            action_net, actions_arr[batch_size * i:batch_size * (i + 1)])
    energies = latent_actions_arr@latent_state
    return actions_arr[np.argmin(energies)]

def F_hash(theta,hyper_params,actions_arr, latent_actions_arr, medians, intervals, projection_vector):
    gym.logger.set_level(40);
    gamma = 1
    env = gym.make(hyper_params.env_name)
    G = 0.0
    done = False
    discount = 1
    state = env.reset()
    state_net = network.itt_get_state_net(theta[hyper_params.action_nn_dim:],hyper_params)
    projection_vector = projection_vector[:,:hyper_params.nA]#last entry is just to normalize actions
    powers = 2 ** np.arange(hyper_params.n_proj_vec)
    while not done:
        latent_state = network.itt_state_forward(state_net, state)
        action = energy_action_hash(latent_actions_arr, actions_arr, latent_state, medians, intervals,
                                       projection_vector,powers)
        state, reward, done, _ = env.step(action)
        G += reward * discount
        discount *= gamma
    return G 


def eval(theta,hyper_params):
    gym.logger.set_level(40)
    env = gym.make(hyper_params.env_name)
    G = 0.0
    done = False
    state = env.reset()
    if hyper_params.useHash==1:
        latent_actions_arr, actions_arr, medians, intervals, projection_vector = utils.construct_hashing_table(
            theta,hyper_params)
        state_net = network.itt_get_state_net(theta[hyper_params.action_nn_dim:],hyper_params)
        projection_vector = projection_vector[:,:hyper_params.nA]
        powers = 2 ** np.arange(hyper_params.n_proj_vec)
        while not done:
            latent_state = network.itt_state_forward(state_net, state)
            action = energy_action_hash(latent_actions_arr, actions_arr, latent_state, medians, intervals,
                                           projection_vector,powers)
            state, reward, done, _ = env.step(action)
            G += reward
    else:
        if hyper_params.model_name=='itt':
            state_net = network.itt_get_state_net(theta[hyper_params.action_nn_dim:], hyper_params)
            action_net = network.itt_get_action_net(theta[:hyper_params.action_nn_dim], hyper_params)
            while not done:
                latent_state = network.itt_state_forward(state_net, state)
                action = energy_action_itt(action_net, latent_state, hyper_params)
                state, reward, done, _ = env.step(action)
                G += reward
        elif hyper_params.model_name=='iot':
            iot_net = network.iot_get_network(theta, hyper_params)
            while not done:
                action = energy_action_iot(iot_net, state, hyper_params)
                state, reward, done, _ = env.step(action)
                G += reward
        else: #explicit model
            explicit_net = network.explicit_get_network(theta, hyper_params)
            while not done:
                action = network.explicit_forward(explicit_net, state)
                state, reward, done, _ = env.step(action)
                G += reward
    return G


def F_itt(theta, hyper_params):
    gym.logger.set_level(40)
    env = gym.make(hyper_params.env_name)
    G = 0.0
    done = False
    discount = 1
    state = env.reset()
    gamma = 1
    state_net = network.itt_get_state_net(theta[hyper_params.action_nn_dim:],hyper_params)
    action_net = network.itt_get_action_net(theta[:hyper_params.action_nn_dim],hyper_params)
    while not done:
        latent_state = network.itt_state_forward(state_net,state)
        action = energy_action_itt(action_net, latent_state,hyper_params)
        state, reward, done, _ = env.step(action)
        G += reward * discount
        discount *= gamma
    return G

def F_iot(theta, hyper_params):
    gym.logger.set_level(40)
    env = gym.make(hyper_params.env_name)
    G = 0.0
    done = False
    discount = 1
    state = env.reset()
    gamma = 1
    iot_net = network.iot_get_network(theta,hyper_params)
    while not done:
        action = energy_action_iot(iot_net,state,hyper_params)
        state, reward, done, _ = env.step(action)
        G += reward * discount
        discount *= gamma
    return G

def F_explicit(theta, hyper_params):
    gym.logger.set_level(40)
    env = gym.make(hyper_params.env_name)
    G = 0.0
    done = False
    discount = 1
    state = env.reset()
    gamma = 1
    explicit_net = network.explicit_get_network(theta,hyper_params)
    while not done:
        action = network.explicit_forward(explicit_net,state)
        state, reward, done, _ = env.step(action)
        G += reward * discount
        discount *= gamma
    return G