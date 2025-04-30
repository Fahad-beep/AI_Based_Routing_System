from .Packets import packet
import builtins
from configparser import ConfigParser

  
configur = ConfigParser()
configur.read(builtins.current_filename)
scale_base_reward = int(configur.get('reward','scale_base_reward'))


class CongestionPoint():
    def __init__(self, x, y):
        self.position = (x,y)
        self.packetRecv = 0
        self.packets_received = []
        self.totalTtl = 0

    def acceptPacket(self, packet):
        self.packetRecv += 1
        self.totalTtl+=packet.get_ttl()
        self.packets_received.append(packet)

    def getReward(self):
        ttl = self.packets_received[-1].get_ttl()
        return scale_base_reward*ttl*ttl

    def getPosition(self):
        return self.position

    def isRouter(self):
        return False
    
    def isCongestionPoint(self):
        return True
     
    def isHost(self):
        return False

    def isBlock(self):
        return False
    
    def getVal(self):
        return self.packetRecv

    def reset(self):
        self.packetRecv = 0
        self.packets_received = []
        self.totalTtl=0

    