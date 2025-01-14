import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as torchF

############################################## itt network

def itt_action_forward_multiple_pass(action_net,actions_arr,hyper_params):
    batch_size = hyper_params.batch_size; n_batch = hyper_params.n_batch
    latent_actions_arr = np.zeros((n_batch*batch_size,hyper_params.nA))
    for i in range(n_batch):
        latent_actions_arr[batch_size*i:batch_size*(i+1)] = itt_action_forward(action_net,actions_arr[batch_size*i:batch_size*(i+1)])
    return latent_actions_arr

class itt_state_tower(nn.Module):
    def __init__(self,state_dim,nA):
        super(itt_state_tower, self).__init__()
        self.fc1 = nn.Linear(state_dim, nA, bias=False)
        self.fc2 = nn.Linear(nA, nA, bias=False)
        self.fc3 = nn.Linear(nA, nA, bias=False)
        self.fc4 = nn.Linear(nA, nA, bias=False)

def itt_state_forward(state_net,state):
    x = (torch.from_numpy(state)).float()
    x = torchF.relu(state_net.fc1(x))
    x = torchF.relu(state_net.fc2(x))
    x = torchF.relu(state_net.fc3(x))
    x = state_net.fc4(x)
    latent_state = x.detach().numpy()
    return latent_state

class itt_action_tower(nn.Module):
    def __init__(self,nA):
        super(itt_action_tower, self).__init__()
        self.fc1 = nn.Linear(nA, nA, bias=False)
        self.fc2 = nn.Linear(nA, nA, bias=False)

def itt_action_forward(action_net,action):
    x = (torch.from_numpy(action)).float()
    x = torchF.relu(action_net.fc1(x))
    x = action_net.fc2(x)
    latent_action = x.detach().numpy()
    return latent_action

def itt_get_state_net(theta,hyper_params):
    state_net = itt_state_tower(hyper_params.state_dim,hyper_params.nA)
    update_nn_params(state_net, theta)
    return state_net

def itt_get_action_net(theta,hyper_params):
    action_net = itt_action_tower(hyper_params.nA)
    update_nn_params(action_net,theta)
    return action_net
    
def get_latent_actions(actions_arr,theta):
    action_net = itt_get_action_net(theta)
    return itt_action_forward(action_net,actions_arr)

############################################## iot network

class iot_tower(nn.Module):
    def __init__(self,state_dim,nA):
        super(iot_tower, self).__init__()
        self.fc1 = nn.Linear(state_dim+nA, nA, bias=False)
        self.fc2 = nn.Linear(nA, nA, bias=False)
        self.fc3 = nn.Linear(nA, nA, bias=False)
        self.fc4 = nn.Linear(nA, nA, bias=False)
        self.fc5 = nn.Linear(nA, nA, bias=False)
        self.fc6 = nn.Linear(nA, 1, bias=False)

def iot_forward(iot_net,state_action):
    x = (torch.from_numpy(state_action)).float()
    x = torchF.relu(iot_net.fc1(x))
    x = torchF.relu(iot_net.fc2(x))
    x = torchF.relu(iot_net.fc3(x))
    x = torchF.relu(iot_net.fc4(x))
    x = torchF.relu(iot_net.fc5(x))
    x = iot_net.fc6(x)
    energy_score = x.detach().numpy()
    return energy_score

def iot_get_network(theta,hyper_params):
    iot_net = iot_tower(hyper_params.state_dim,hyper_params.nA)
    update_nn_params(iot_net, theta)
    return iot_net

############################################## iot network

class explicit_tower(nn.Module):
    def __init__(self,state_dim,nA):
        super(explicit_tower, self).__init__()
        self.fc1 = nn.Linear(state_dim, nA, bias=False)
        self.fc2 = nn.Linear(nA, nA, bias=False)
        self.fc3 = nn.Linear(nA, nA, bias=False)
        self.fc4 = nn.Linear(nA, nA, bias=False)
        self.fc5 = nn.Linear(nA, nA, bias=False)
        self.fc6 = nn.Linear(nA, nA, bias=False)

def explicit_forward(explicit_net,state):
    x = (torch.from_numpy(state)).float()
    x = torchF.relu(explicit_net.fc1(x))
    x = torchF.relu(explicit_net.fc2(x))
    x = torchF.relu(explicit_net.fc3(x))
    x = torchF.relu(explicit_net.fc4(x))
    x = torchF.relu(explicit_net.fc5(x))
    x = explicit_net.fc6(x)
    action = x.detach().numpy()
    action = np.tanh(action)
    return action

def explicit_get_network(theta,hyper_params):
    explicit_net = explicit_tower(hyper_params.state_dim,hyper_params.nA)
    update_nn_params(explicit_net, theta)
    return explicit_net


############################################## network helper functions

def get_theta_dim(model_name,state_dim,nA):
    if model_name == 'itt':
        state_net = itt_state_tower(state_dim,nA)
        action_net = itt_action_tower(nA)
        state_nn_dim = get_nn_dim(state_net)
        action_nn_dim = get_nn_dim(action_net)
        return action_nn_dim+state_nn_dim,action_nn_dim,state_nn_dim
    elif model_name == 'iot':
        iot_net = iot_tower(state_dim,nA)
        iot_theta_dim = get_nn_dim(iot_net)
        return iot_theta_dim,0,iot_theta_dim
    else: #explicit
        explicit_net = explicit_tower(state_dim,nA)
        explicit_theta_dim = get_nn_dim(explicit_net)
        return explicit_theta_dim,0,explicit_theta_dim

def update_nn_params(input_nn,new_params):
    params = list(input_nn.parameters())
    current_index= 0
    for i in range(len(params)):
        shape = params[i].data.detach().numpy().shape
        arr = new_params[current_index:current_index+np.product(shape)].reshape(shape)
        params[i].data = (torch.from_numpy(arr)).float()
        current_index+=np.product(shape)
    for param in params:
        param.requires_grad = False

def get_nn_dim(input_nn):
    params = list(input_nn.parameters())
    counter = 0
    for i in range(len(params)):
        shape = params[i].data.detach().numpy().shape
        counter+=np.product(shape)
    return counter