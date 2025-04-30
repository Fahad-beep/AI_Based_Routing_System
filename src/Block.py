from torch import true_divide

class Block():
    def __init__(self,position):
        self.position= position

    def isBlock(self):
        return True
    
    def reset(self):
        pass

    def isRouter(self):
        return False
    
    def isCongestionPoint(self):
        return True
     
    def isHost(self):
        return False