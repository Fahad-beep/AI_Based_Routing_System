# MARL-Packet-Router: Intelligent Network Routing with Multi-Agent Reinforcement Learning

## Project Overview

### Objective

Develop an adaptive packet routing system using Multi-Agent Reinforcement Learning (MARL) that optimizes network performance by:

- Reducing congestion
- Minimizing latency
- Improving packet delivery efficiency

## Key Features

- **Dynamic Learning**: Routers adapt routing decisions based on real-time network conditions
- **Collaborative Agents**: Multiple routers work together to optimize packet routing
- **Scalable Architecture**: Designed to handle varying network sizes and topologies

---

## System Architecture

### Network Components

- **Routers (Agents)**: Autonomous nodes that learn optimal routing strategies
- **Host**: Destination node for all packets
- **Congestion Points**: Special nodes that simulate network bottlenecks
- **Blocks**: Unreachable nodes that constrain routing paths

### Map Representation

Example 4x3 grid:
R C R
R X R
R R R
H X X

- `R` = Router (Agent)
- `C` = Congestion point
- `X` = Block (unreachable)
- `H` = Host (destination)

---

## Installation & Setup

### Requirements

- Python 3.8+
- PyTorch 1.11.0
- NumPy 1.22.3
- Matplotlib 3.5.2
- tqdm 4.64.0

### Installation Steps

```bash
# Create virtual environment
python -m venv marl_env
source marl_env/bin/activate  # Linux/Mac
marl_env\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

or
# Setup everything by MakeFile
make setup
```

## Running

```
mkdir <any folder name>
copy config.ini <folder name>
update makefile's CONFIG_DIR to <created folder>
make run
```
