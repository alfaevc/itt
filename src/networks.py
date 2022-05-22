import torch.nn as nn


class state_tower(nn.Module):
    def __init__(self, state_dim, nA):
        super(state_tower, self).__init__()
        self.fc1 = nn.Linear(state_dim, nA, bias=False)  
        self.fc2 = nn.Linear(nA, nA, bias=False)
        self.fc3 = nn.Linear(nA, nA, bias=False)
        self.fc4 = nn.Linear(nA, nA, bias=False)
    
class action_tower(nn.Module):
    def __init__(self, nA):
        super(action_tower, self).__init__()
        self.fc1 = nn.Linear(nA, nA, bias=False)
        self.fc2 = nn.Linear(nA, nA, bias=False)

class one_tower(nn.Module):
    def __init__(self, state_dim, nA):
        super(state_tower, self).__init__()
        self.fc1 = nn.Linear(state_dim+nA, nA, bias=False)  
        self.fc2 = nn.Linear(nA, nA, bias=False)
        self.fc3 = nn.Linear(nA, nA, bias=False)
        self.fc4 = nn.Linear(nA, nA, bias=False)
        # self.fc5 = nn.Linear(nA, nA, bias=False)
        # self.fc6 = nn.Linear(nA, 1, bias=False)

  