from src.DQN.dqn_agent import DQNAgent
from .utils import getNetworkDistance
import math
from matplotlib import pyplot as plt
from configparser import ConfigParser
import builtins
import random

configur = ConfigParser()
configur.read(builtins.current_filename)
maxTtl = int(configur.get('packet','maxTtl')) 
defaultTtl = int(configur.get('packet','def_ttl')) 
router_to_router_scale = float(configur.get('reward','router_to_router_scale'))
packet_drop_reward = int(configur.get('reward','packet_drop_reward'))
ttl_zero_reward = int(configur.get('reward','ttl_zero_reward'))
agent_to_agent_scale = float(configur.get('reward','agent_to_agent_scale'))
scaling_type = configur.get('scaling_factor','type')
include_distance = configur.getboolean('reward','include_distance')

class Router():
    def __init__(self, neighbours, x, y, CongestionPoint, batchsize=64):
        
        self.batchsize = batchsize
        self.state_size = 1
        self.targetCongestionPoint = CongestionPoint
        self.queue = []
        self.neighbours = neighbours
        self.dqn_object = None
        self.action_size = 1
        self.latest_loss = 0
        self.losses = []
        self.position = (x,y)
        self.latest_queue = []
        self.q_values = []

    def getCurrentState(self):
        state = [len(self.queue)]
        for neighbour in self.neighbours:
            if not neighbour.isCongestionPoint():
                state.append(len(neighbour.queue))
            else:
                state.append(0)
        return state

    def initDQN(self,device):
        self.dqn_object = DQNAgent(device ,self.state_size, self.action_size)

    def loadModel(self, filename):
        print(f"\n[Router at {self.position}] Loading model from: {filename}")
        self.dqn_object.loadModel(filename)
        print(f"[Router at {self.position}] Model loaded successfully")

    def pushQueue(self, packet):
        self.queue.append(packet)
        return True
    
    def popQueue(self):
        if len(self.queue) == 0 :
            return -1
        return self.queue.pop(0)

    def getTopPacket(self):
        if len(self.latest_queue) == 0:
            return -1
        return self.latest_queue[-1]

    def nextAction(self,state):
        return self.dqn_object.selectAction(state)

    def trainAgent(self,state,action,nextState,reward):
        self.dqn_object.memory.store(state=state, action=action, next_state=nextState, reward=reward)
        self.latest_loss = self.dqn_object.learn(batchsize=self.batchsize)

    def saveLoss(self):
        self.losses.append(self.latest_loss)

    def getLoss(self):  
        return self.losses

    def acceptPacket(self,packet):
        self.latest_queue.append(packet)

    def getPosition(self):
        return self.position

    def addNeighbour(self,neighbour):
        self.neighbours.append(neighbour)
        self.action_size+=1
        self.state_size+=1


    def isRouter(self):
        return True
    
    def isCongestionPoint(self):
        return False
     
    def isHost(self):
        return False
    
    def isBlock(self):
        return False
    
    def getReward(self):       
        top_packet_ttl = self.getTopPacket().get_ttl()    
        reward = router_to_router_scale*self.dqn_object.getQValue(self.getCurrentState())
        if scaling_type == 'square':
            scaled_reward = reward*((top_packet_ttl/defaultTtl)**2)
        elif scaling_type == 'exponential':
            scaled_reward = reward*math.exp(top_packet_ttl-defaultTtl)     
        elif scaling_type == 'fraction': 
            scaled_reward = reward*(top_packet_ttl/defaultTtl)
        else:
            scaled_reward = reward
        return scaled_reward

    def run(self, train = True):
        state = self.getCurrentState()
        for packet in self.queue:  
            packet.decrease_ttl()
        topPacket = self.popQueue()

        if topPacket == -1:
            return
        
        nextAction = self.nextAction(state)
        self.q_values.append(self.dqn_object.getQValue(state))

        if not train:
            print("Position : ",self.getPosition())
            print("States : ", state)
            print("TTL : ", topPacket.get_ttl())
            if(nextAction == len(self.neighbours)):
                print("Packet dropped")
            else:
                print("Packet forwarded to neighbour : ",self.neighbours[nextAction].getPosition())
            print("Q-Value : ", self.dqn_object.getQValue(state))
            plt.plot(self.q_values)
            plt.savefig(f'./agent_at_{self.position}.png')
            plt.close()

        nextState = self.getCurrentState()
        if topPacket.get_ttl() <= 0:
            if train:
                self.trainAgent(state,nextAction,nextState,ttl_zero_reward) 
            return
        
        if  nextAction == len(self.neighbours):
            if train:  
                self.trainAgent(state,nextAction,nextState,packet_drop_reward) 
            return
        
        self.neighbours[nextAction].acceptPacket(topPacket)
        nextState = self.getCurrentState()
        nextState[nextAction+1]+=1
        reward = self.neighbours[nextAction].getReward()
        if(include_distance):
            reward *= 1/getNetworkDistance(self.getPosition(), self.targetCongestionPoint.getPosition())

        if train:
            self.trainAgent(state,nextAction,nextState,reward) 


    def randomRun(self):
        state = self.getCurrentState()
        for packet in self.queue:  
            packet.decrease_ttl()
        topPacket = self.popQueue()

        if topPacket == -1:
            return

        action = random.randint(0,len(self.neighbours))
        if(topPacket.get_ttl() <= 0):
            reward = ttl_zero_reward
            nextState = self.getCurrentState()
            self.dqn_object.memory.store(state=state, action=action, next_state=nextState, reward=reward)
        
        if action == len(self.neighbours):
            reward = packet_drop_reward
            nextState = self.getCurrentState()
            self.dqn_object.memory.store(state=state, action=action, next_state=nextState, reward=reward)
        
        else:    
            self.neighbours[action].acceptPacket(topPacket)  
            reward = self.neighbours[action].getReward()
            
            if(include_distance):
                reward *= 1/getNetworkDistance(self.getPosition(), self.targetCongestionPoint.getPosition())

            nextState = self.getCurrentState()
            nextState[action+1] += 1
            self.dqn_object.memory.store(state=state, action=action, next_state=nextState, reward=reward)

    
    def getVal(self):
        return len(self.queue)

    def reset(self):
        self.queue = []

    def update_state(self):
        for packets in self.latest_queue:
            self.queue.append(packets)
        self.latest_queue = []
