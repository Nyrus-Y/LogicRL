# LogicRL
This is the code for RL agents using logic.
![](image/LogicRL.png)

## Installation

```bash
pip install -r requirements.txt
```
from the nsfr folder:
```bash
pip install -e . # installs in dev mode
```

## How to use


**Example to play with a trained ppo agent**
```
python3 play.py -s 0 -alg ppo -m coinjump -env CoinJumpEnv-v1   
```  
The trained model can be found in folder: _models/coinjump_ or _models/bigfish_

**Example to train an logic agent for coinjump env using 'coinjump_human_assisted' rules.**
```
python3 train.py -s 0 -alg logic -m coinjump -env CoinJumpEnv-v1  -r 'coinjump_human_assisted'
```

Description of Args

* **--algorithm -alg**:

The algorithm to use for playing or training, choice: _ppo_, _logic_.

* **--mode -m**:

Game mode to play or train with, choice: _coinjump_, _bigfish_, _heist_.

* **--environment -env**:

the specific environment for playing or training,

_CoinJumpEnv-v1_ contains key, door and one enemy.  
_CoinJumpEnv-v2_ has one more enemy.  
_bigfishm_  contains one bigger fish and one smaller fish.  
_bigfishc_  contains one red fish and one green fish. agent need to avoid red fish and eat green fish.  
_eheistc1_  contains 2 pairs of key and door.  
_eheistc2_  contains 2 pairs of key and door with different color.  
_eheist_    contains 3 pairs of key and door.  
* **--rules -r**:

_rules_ is required when train logic agent.

Logic agent require a set of data which provide the first order logic rules.  

e.g. '_coinjump_human_assisted_' indicate the rules is human-generated.

So for new rules, just need to be added to the choice of argument '--rules' and dataset.  

dataset can be found in folder: _src/nsfr/data_

'--rules' is also for some situation like using reward shaping:  

e.g. 'ppo_simple_policy' can be helpful when train ppo agent of coinjump

Models will be saved to folder: src/checkpoints  
Models that use to run should be moved to folder: src/models

**Using Beam Search to find a set of rules**

![](image/beam_search.png)
With scoring:
```
python3 beam_search.py -m coinjump -r coinjump_root -t 3 -n 8 --scoring True -d coinjump.json  
``` 

Without scoring:
``` 
python3 beam_search.py -m bigfish -r bigfishm_root -t 3 -n 8 
``` 
* **--t**:  Number of rule expansion of clause generation.
* **--n**:  The size of the beam.
* **--scoring**: To score the searched rules, a dataset of states information is required.
* **-d**: The name of dataset to be used for scoring.

