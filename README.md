# Use-Knowledge-Representation-and-Reasoning-for-the-policy


## Installation
The packages are specified in [requirements.txt](./requirements.txt). Please install the packages by:
```
pip install -r requirements.txt
```

### PLAY with trained PPO agent
key_door:
1. run src/coinjump_play_KD 
2. input name of trained model: PPO_Key_Door.pth   

dodge_enemy:
1. run src/coinjump_play_D 
2. input name of trained model: PPO_Dodge.pth   

enemy and key_door:
1. run src/coinjump_play_V1 
2. input name of trained model: PPO_V1_enemy_door.pth 

### PLAY with FOL

run src/coinjump_V1.py 
or  src/coinjump_D.py
or  src/coinjump_KD.py

### something else:
here to choose model:

path of models: src/ppo_coinjump_model


here to define FOL:

path of clasues: data/lang/coinjump
