# Use-Knowledge-Representation-and-Reasoning-for-the-policy


## Installation
The packages are specified in [requirements.txt](./requirements.txt). Please install the packages by:
```
pip install -r requirements.txt
```

### PLAY with trained PPO agent
Run the file in src/CoinJump/:

key_door:
1. **run** coinjump_play_KD 
2. input model: PPO_Key_Door.pth   

dodge_enemy:
1. **run** coinjump_play_D 
2. input model: PPO_Dodge.pth   

enemy and key_door:
1. **run** coinjump_play_V1 
2. input model: PPO_V1_enemy_door.pth 

### PLAY with pure Logic Policy

env include enemy,key and door: 

**run** src/CoinJump/coinjump_play_V1_logic.py 

env include enemy:

**run** src/CoinJump/coinjump_D_logic.py

env include key and door :

**run** src/CoinJump/coinjump_KD_logic.py


### PLAY with trained Logic Policy
env include enemy,key and door: 

1. **run** src/CoinJump/coinjump_play_trained_LP.py
2. input model coinjump_LP_10clauses.pth


### something else:
here to choose model:

path of models: src/ppo_coinjump_model


here to define FOL:

path of clasues: data/lang/coinjump
