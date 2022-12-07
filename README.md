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

Description of Args

* **--algorithm -alg**:

The algorithm to use for playing or training, choice: _ppo_, _logic_.

* **--mode -m**:

Game mode to play or train with, choice: _coinjump_, _bigfish_, _heist_.

* **--environment -env**:

the specific environment for playing or training,

e.g. _CoinJumpEnvNeural-v0_ is use to train neural agent of ppo contains key,door and enemy.

_bigfishm_  contains one bigger fish and one smaller fish.  
_bigfishc_  contains one red fish and one green fish. agent need to avoid red fish and eat green fish.

* **--rules -r**:

_rules_ is required when train logic agent.

Logic agent require a set of data which provide the first order logic rules.  

e.g. '_coinjump_5a_' indicate the rules with 5 clauses.

So for new rules, just need to be added to the choice of argument '--rules' and dataset.  

dataset can be found in folder: _src/nsfr/data_

'--rules' is also for some situation like using reward shaping:  

e.g. 'ppo_simple_policy' can be helpful when train ppo agent of coinjump




**Example to play with a trained ppo agent**
```
python3 play.py -s 0 -alg ppo -m coinjump -env CoinJumpEnvNeural-v0  
```  
The trained model can be found in folder: _models/coinjump_ or _models/bigfish_

**Example to train an logic agent for coinjump env using 'coinjump_5a' rules.**
```
python3 train.py -s 0 -alg logic -m coinjump -env CoinJumpEnvLogic-v0  -r 'coinjump_5a'
```

model will be saved to folder: checkpoints

## TO BE DONE
heist
plot  
a formal README...   
beam search

## Contributing

## License

[MIT](https://choosealicense.com/licenses/mit/)
