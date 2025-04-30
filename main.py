import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from configparser import ConfigParser
import builtins

def debug_print(*args, **kwargs):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}]", *args, **kwargs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
debug_print(f"Using device: {device}")

torch.cuda.empty_cache()
if(len(sys.argv) <= 1):
    debug_print("Please provide the config file folder name")
    exit()

folder_name = sys.argv[1]
debug_print(f"Using configuration from folder: {folder_name}")

builtins.current_filename = "{}/config.ini".format(folder_name)
configur = ConfigParser()
configur.read(builtins.current_filename)


debug_print("\nConfiguration Parameters:")
for section in configur.sections():
    debug_print(f"[{section}]")
    for key, val in configur.items(section):
        debug_print(f"  {key} = {val}")

num_memory_fill_eps = int(configur.get('train_model','num_memory_fill_eps'))
tot_episodes = int(configur.get('train_model','tot_episodes'))
tot_time = int(configur.get('train_model','tot_time'))
update_frequency = int(configur.get('train_model','update_frequency'))
save_frequency = int(configur.get('train_model','save_frequency'))
generate_packets_till = int(configur.get('test_model','generate_packets_till'))
gap_time= int(configur.get('train_model','gap_time'))

n = int(configur.get('map','n'))
m = int(configur.get('map','m'))
p = float(configur.get('map','p'))
map_name = configur.get('map','name')

debug_print(f"\nInitializing {n}x{m} map from file: {map_name}")

from src.Map import Map
map_ = Map(n,m,p)
map_.read()

debug_print("\nInitial Network Configuration:")
debug_print(f"Number of Hosts: {len(map_.getHosts())}")
debug_print(f"Number of Routers: {len(map_.getRouters())}")
debug_print(f"Congestion Point Position: {map_.getCongestionPoint().getPosition()}")

Hosts = map_.getHosts()
CongestionPoint = map_.getCongestionPoint()
Routers = map_.getRouters()

def fillMemory():
    debug_print(f"\nFilling replay memory with {num_memory_fill_eps} episodes")
    for ep in range(num_memory_fill_eps):
        debug_print(f"\nMemory Fill Episode {ep+1}/{num_memory_fill_eps}")
        for time in range(tot_time):
            if(time%gap_time==0):
                for host in Hosts:
                    host.run()
                    debug_print(f"Time {time}: Host at {host.getPosition()} generated packet", end='\r')

            for router in Routers:
                router.randomRun()
                debug_print(f"Time {time}: Router at {router.getPosition()} took random action", end='\r')

            for router in Routers:
                router.update_state()

        map_.resetAll()
        debug_print(f"Completed Memory Fill Episode {ep+1}. Reset all network nodes.")

def train(foldername,graphics=False):
    debug_print("\nStarting Training Phase")
    step_cnt = 0

    for episode in tqdm(range(tot_episodes), position=0, leave=True):
        debug_print(f"\nEpisode {episode+1}/{tot_episodes}")
        
        if step_cnt % update_frequency == 0 and step_cnt!=0:
            debug_print(f"Updating target networks for all routers at step {step_cnt}")
            for router in Routers:
                router.dqn_object.updateTargetNet()

        episode_packets = 0
        for time in range(tot_time):
            if(time%gap_time==0):
                for host in Hosts:
                    host.run()
                    debug_print(f"Time {time}: Host at {host.getPosition()} generated packet", end='\r')

            for router in Routers:
                router.run()
                debug_print(f"Time {time}: Router at {router.getPosition()} took learned action", end='\r')

            for router in Routers:
                router.update_state()

            if graphics and episode == tot_episodes-1:
                map_.renderMap()

        step_cnt += 1
        episode_packets = CongestionPoint.packetRecv
        debug_print(f"Episode {episode} completed. Packets received: {episode_packets}")
        
        for router in Routers:
            router.dqn_object.updateEpsilon()
            router.saveLoss()
            loss_info = router.latest_loss if router.latest_loss is not None else "N/A"
            debug_print(f"Router at {router.getPosition()} - Epsilon: {router.dqn_object.epsilon:.4f}, Loss: {loss_info}")

        map_.resetAll()

        if(episode% save_frequency == 0):
            debug_print(f"Saving models at episode {episode}")
            for router in Routers:
                save_path = './{}/router_at_{}'.format(foldername,router.getPosition())
                print(f"\n[Training] Saving model for router at {router.getPosition()} to: {save_path}")
                router.dqn_object.saveModel(save_path)
                debug_print(f"Saved model for router at {router.getPosition()} to {save_path}")
                print(f"[Training] Model saved successfully")

def test(folder_name,render=True):
    debug_print("\nStarting Testing Phase")
    debug_print(f"Will generate packets till time step {generate_packets_till}")
    
    map_.resetAll()
    CongestionPoint.reset()
    
    for router in Routers:
        router.dqn_object.turn_off_exploration()
        debug_print(f"Router at {router.getPosition()} exploration turned off")

    step_cnt = 0
    num_packets=[]
    total_ttl=[]
    time_steps=[]
    t=0
    
    debug_print("Beginning test episode...")
    while True:
        if(t%gap_time==0):
            step_cnt += 1
            if step_cnt <= generate_packets_till:
                for host in Hosts:
                    host.run()
                    debug_print(f"Time {t}: Host at {host.getPosition()} generated packet", end='\r')

        for router in Routers:
            router.run(False)
            debug_print(f"Time {t}: Router at {router.getPosition()} took action", end='\r')

        for router in Routers:
            router.update_state()

        if render:
            map_.renderMap()

        num_packets.append(CongestionPoint.packetRecv)
        total_ttl.append(CongestionPoint.totalTtl)
        time_steps.append(t)
        t+=1
           
        end = True
        for router in Routers:
            if router.getVal() != 0:
                end = False
                break

        for host in Hosts:
            if host.getQueueSize() !=0:
                end = False
                break

        if end:
            debug_print(f"\nTest completed at time {t}. Total packets received: {CongestionPoint.packetRecv}")
            break
    
    os.makedirs("{}/Plots".format(folder_name), exist_ok=True)
    debug_print("Saving test result plots...")
    plt.plot(time_steps,num_packets , color ='blue', label ='Packets Received')
    plt.savefig('{}/Plots/Packet_Received.png'.format(folder_name))
    plt.close()

    plt.plot(time_steps,total_ttl , color ='blue', label ='Sum of TTL')
    plt.savefig('{}/Plots/SumOfTtl.png'.format(folder_name))
    plt.close()

def meanTtl():
    packets = map_.getCongestionPoint().packets_received
    if len(packets)==0:
        debug_print("No packets received at congestion point!")
        return -1
    avg = sum([packet.get_ttl() for packet in packets])/len(packets)
    debug_print(f"Average TTL of received packets: {avg:.2f}")
    return avg

def generatePlot(folder_name):
    debug_print("\nGenerating training plots...")
    os.makedirs("{}/Plots".format(folder_name), exist_ok=True)
    for router in Routers:
        loss = router.getLoss()
        epi_list = list(range(1,len(loss)+1))
        plt.plot(epi_list, loss, color ='orange', label ='Router Loss')
        plot_path = '{}/Plots/router_at_{}.png'.format(folder_name,router.getPosition())
        plt.savefig(plot_path)
        plt.close()
        debug_print(f"Saved loss plot for router at {router.getPosition()} to {plot_path}")


if __name__ ==  '__main__':
    debug_print("\nStarting main execution")
    os.makedirs("{}/model_parameters".format(folder_name), exist_ok=True)
    
    debug_print("Initializing router models...")
    map_.initModels(device)
    
    if configur.get('train_model','train') == 'True':
        debug_print("\nTraining mode activated")
        fillMemory()
        train("{}/model_parameters".format(folder_name),False)
    else:
        debug_print("\nSkipping training (train=False in config)")
    
    debug_print("Loading models for testing...")
    model_dir = os.path.join("config", "model_parameters")
    print(f"\n[Main] Loading models from: {os.path.abspath(model_dir)}")
    map_.loadModel(model_dir)
    
    debug_print("Running test...")
    test(folder_name)
    
    debug_print("Generating plots...")
    generatePlot(folder_name)
    
    debug_print("\nFinal Results:")
    print('Mean TTL of all packets received by congestion point: ',meanTtl())
    debug_print("Execution completed")