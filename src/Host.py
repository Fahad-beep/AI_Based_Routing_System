import sys
from .Packets import packet
import numpy as np
from configparser import ConfigParser
import builtins
from .Router import Router


configur = ConfigParser()
configur.read(builtins.current_filename)
sys.path.append(".")


class Host():
    def __init__(self, rate, def_ttl, x, y, transmission_rate=10):
        self.rate = rate
        self.def_ttl = def_ttl
        self.position = (x,y)
        self.neighbours = []
        self.total_packets = 0
        self.queue = []
    
    def generatePacket(self):
        num_packets = self.rate
        self.total_packets += num_packets
        for i in range(num_packets):
            self.queue.append(packet(self.def_ttl))
    
    def findNeighbour(self):
        for router in self.neighbours:
            if router.isCongestionPoint():
                return router
        queues = np.array([agent.getVal() for agent in self.neighbours])
        agents = [agent for agent in self.neighbours]
        return agents[ np.argmin(queues) ]

    def isRouter(self):
        return False
    
    def isCongestionPoint(self):
        return False
     
    def isHost(self):
        return True

    def isBaseStation(self):
        return False

    def isBlock(self):
        return False
    
    def run(self):
        self.generatePacket()
        for i in range(self.rate):
            if len(self.neighbours)==0:
                continue
            if self.getQueueSize() > 0:
                packet = self.queue.pop(0)
                agent=self.findNeighbour()
                agent.acceptPacket(packet)

    def getPosition(self):
        return self.position

    def addNeighbour(self,neighbour: Router):
        self.neighbours.append(neighbour)

    def getVal(self):
        return self.total_packets

    def getQueueSize(self):
        return len(self.queue)

    def reset(self):
        self.total_packets=0
        self.queue = []

