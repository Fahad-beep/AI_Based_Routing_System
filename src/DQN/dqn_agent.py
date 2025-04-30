import random

import torch
import torch.nn.functional as F
from .dqn_network import DQNNet
from .buffer_replay import ReplayMemory
from configparser import ConfigParser
import builtins
  
configur = ConfigParser()
configur.read(builtins.current_filename)

discount1 = float(configur.get('architecture','discount'))
eps_max1= float(configur.get('architecture','eps_max'))
eps_min1= float(configur.get('architecture','eps_min'))
eps_decay1 = float(configur.get('architecture','eps_decay'))
memory_capacity1 = int(configur.get('architecture','memory_capacity'))
lr1= float(configur.get('architecture','lr'))

class DQNAgent:
    def __init__(self, device, state_size, action_size, 
                    discount=discount1,
                    eps_max=eps_max1,
                    eps_min=eps_min1,
                    eps_decay=eps_decay1,
                    memory_capacity=memory_capacity1,
                    lr=lr1,
                    train_mode=True):

        self.device = device
        self.epsilon = eps_max
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay

        self.discount = discount

        self.state_size = state_size
        self.action_size = action_size

        self.policy_net = DQNNet(self.state_size, self.action_size, lr).to(self.device)
        self.target_net = DQNNet(self.state_size, self.action_size, lr).to(self.device)
        self.target_net.eval()
        
        self.memory = ReplayMemory(capacity=memory_capacity)

    def selectAction(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action = self.policy_net.forward(state)
        return torch.argmax(action).item()

    def turn_off_exploration(self):
        self.epsilon =  0

    def updateTargetNet(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def updateEpsilon(self):
        
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)


    def selectAction(self, state):

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action = self.policy_net.forward(state)
        return torch.argmax(action).item()


    def learn(self, batchsize):
        if len(self.memory) < batchsize:
            return
        states, actions, next_states, rewards = self.memory.sample(batchsize, self.device)
        q_pred = self.policy_net.forward(states).gather(1, actions.view(-1, 1)) 
        
        
        q_target = self.target_net.forward(next_states).max(dim=1).values
        y_j = rewards + (self.discount * q_target)
        y_j = y_j.view(-1, 1)

        self.policy_net.optimizer.zero_grad()
        loss = F.mse_loss(y_j, q_pred).mean()
        loss.backward()
        self.policy_net.optimizer.step()
        return float(loss)
        

    def saveModel(self, filename):
        self.policy_net.saveModel(filename)

    def loadModel(self, filename):
        self.policy_net.loadModel(filename=filename, device=self.device)

    def getQValue(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            q_values = self.target_net.forward(state)

        max_q_value = float(torch.max(q_values, dim=1)[0])

        return max_q_value











