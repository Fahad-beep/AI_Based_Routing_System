from .Host import Host
from .CongestionPoint import CongestionPoint
from .Router import Router
from .Block import Block
from configparser import ConfigParser
import os
import builtins
configur = ConfigParser()
configur.read(builtins.current_filename)

defTtl = int(configur.get('packet','def_ttl'))

class Map:
    def __init__(self, n, m, p):
        self.n = n
        self.m = m
        self.p = p
        self.grid = []
        self.routers = []
        self.hosts = []
        self.congestion_point = None
        self.blocks = []

    def initModels(self, device):
        for agent in self.routers:
            print(f"\n[Initialization] Creating DQN model for router at {agent.getPosition()}")
            agent.initDQN(device)
            print(f"[Initialization] Model created successfully")

    def renderMap(self):
        print("\nCurrent Network State:")
        for i in range(self.n):
            row = []
            for j in range(self.m):
                node = self.grid[i][j]
                if node.isBlock():
                    row.append(" X ")
                elif node.isCongestionPoint():
                    row.append(f"[{node.packetRecv}]")
                elif node.isRouter():
                    row.append(f" {node.getVal()} ")
                elif node.isHost():
                    row.append(f"({node.getVal()})")
                else:
                    row.append("   ")
            print("|" + "|".join(row) + "|")
            print("-" * (self.m * 4 + 1))

    def read(self):
        file = open('./Maps/' + configur.get('map','name'), 'r')
        self.grid = [[None for _ in range(self.m)] for _ in range(self.n)]
        
        for i, line in enumerate(file):
            j = 0
            for char in line:
                if j >= self.m:
                    break
                if char == ' ':
                    continue
                position = (i, j)
                if char == 'C':
                    self.grid[i][j] = CongestionPoint(i, j)
                    self.congestion_point = self.grid[i][j]
                elif char == 'R':
                    router = Router([], i, j, None)
                    self.grid[i][j] = router
                    self.routers.append(router)
                elif char == 'H':
                    host = Host(10, defTtl, i, j)
                    self.grid[i][j] = host
                    self.hosts.append(host)
                elif char == 'X':
                    block = Block(position)
                    self.grid[i][j] = block
                    self.blocks.append(block)
                j += 1 

        for router in self.routers:
            router.targetCongestionPoint = self.congestion_point
                

        for i in range(self.n):
            for j in range(self.m):
                node = self.grid[i][j]
                if node is None or node.isBlock():
                    continue
                
                neighbors = []
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < self.n and 0 <= nj < self.m:
                        neighbor = self.grid[ni][nj]
                        if neighbor and not neighbor.isBlock() and (neighbor.isRouter() or neighbor.isCongestionPoint()):
                            neighbors.append(neighbor)
                
                if node.isRouter() or node.isHost():
                    for neighbor in neighbors:
                        node.addNeighbour(neighbor)

    def getCongestionPoint(self):
        return self.congestion_point

    def getRouters(self):
        return self.routers

    def getHosts(self):
        return self.hosts
    
    def loadModel(self, folder_name):
        print(f"\n[Map] Loading models from folder: {folder_name}")
        for router in self.routers:
            i, j = router.getPosition()
            model_path = os.path.join(os.getcwd(), folder_name, f"router_at_({i}, {j})")
            print(f"[Map] Attempting to load from absolute path: {os.path.abspath(model_path)}")
            if not os.path.exists(model_path):
                print(f"[ERROR] File not found at: {os.path.abspath(model_path)}")
                print(f"[DEBUG] Current working directory: {os.getcwd()}")
                print(f"[DEBUG] Directory contents: {os.listdir(os.path.join(os.getcwd(), folder_name))}")
                continue
                
            router.loadModel(model_path)
        print("[Map] Model loading completed")

    def renderNetwork(self):
        print("\nCurrent Network State:")
        for i in range(self.n):
            row = []
            for j in range(self.m):
                node = self.grid[i][j]
                if node.isBlock():
                    row.append(" X ")
                elif node.isCongestionPoint():
                    row.append(f"[{node.packetRecv}]")
                elif node.isRouter():
                    row.append(f" {node.getVal()} ")
                elif node.isHost():
                    row.append(f"({node.getVal()})")
                else:
                    row.append("   ")
            print("|" + "|".join(row) + "|")
            print("-" * (self.m * 4 + 1))

    def resetAll(self):
        for i in range(self.n):
            for j in range(self.m):
                self.grid[i][j].reset()


    def resetNetwork(self):
        self.congestion_point.reset()
        for router in self.routers:
            router.reset()
        for host in self.hosts:
            host.reset()

    def initializeRouterModels(self, device):
        for router in self.routers:
            router.initDQN(device)

    def loadRouterModels(self, folder_name):
        for router in self.routers:
            i, j = router.getPosition()
            model_path = f"./{folder_name}/router_at_({i},{j})"
            router.loadModel(model_path)