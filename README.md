# LogicRL


## Installation

```bash
pip install requirements.txt
```

## How to use

Description About Args

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

Logic agent require a set of data which provide the first order logic rules. e.g. '_coinjump_5a_' indicate the rules with 5 clauses.

Dataset can be found in path: _src/nsfr/data_
  

**Example to play with a trained ppo agent**
```
python3 play.py -s 0 -alg ppo -m coinjump -env CoinJumpEnvNeural-v0  
```  
The trained model can be found in path: _models/coinjump_ or _models/bigfish_

**Example to train an logic agent for coinjump env using 'coinjump_5a' rules.**
```
python3 train.py -s 0 -alg logic -m coinjump -env CoinJumpEnvLogic-v0  -r 'coinjump_5a'
```

## Contributing

## License

[MIT](https://choosealicense.com/licenses/mit/)
