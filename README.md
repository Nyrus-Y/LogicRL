# Use-Knowledge-Representation-and-Reasoning-for-the-policy


## Installation
The packages are specified in [requirements.txt](./requirements.txt). Please install the packages by:
```
pip install -r requirements.txt
```

### PLAY
key_door:
1. run src/coinjump_play_KD 
2. input name of trained model: PPO_key_door.pth   

dodge_enemy:
1. run src/coinjump_play_D 
2. input name of trained model: PPO_enemy.pth   

enemy and key_door:
1. run src/coinjump_play_V1 
2. input name of trained model: PPO_V1_enemy_door.pth 

here to choose model:
path of models: src/ppo_coinjump_model

here to define FOL of the game:
path of clasues: data/lang/coinjump  
(coinjump1 is for testing)
