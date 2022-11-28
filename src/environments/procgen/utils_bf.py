import torch
import numpy as np


def prediction_to_action_bf(prediction, args, prednames=None):
    if args.alg == 'ppo':
        if not isinstance(prediction, int):
            #prediction = torch.argmax(prediction).cpu().numpy().reshape(-1)
            prediction = torch.argmax(prediction)
        action = simplify_action(prediction)
    elif args.alg == 'logic':
        # print_explaining(prediction, prednames)
        action = num_action_bf(prediction)
    return action


def extract_reasoning_state(states, env="size"):
    """
    states = [x, y, radius]
    extracted_states = [agent, fish, r, x, y]
    size: 3*5 3 fish, 5 features
    """
    states = torch.from_numpy(states).squeeze()
    if env == "size":
        extracted_state = torch.zeros((3, 5))
        for i, state in enumerate(states):
            if i == 0:
                extracted_state[i, 0] = 1  # agent
                extracted_state[i, 2] = states[i, 2]  # radius
                extracted_state[i, 3] = states[i, 0]  # X
                extracted_state[i, 4] = states[i, 1]  # Y
            else:
                extracted_state[i, 1] = 1  # fish
                extracted_state[i, 2] = states[i, 2]  # radius
                extracted_state[i, 3] = states[i, 0]  # X
                extracted_state[i, 4] = states[i, 1]  # Y

        extracted_state = extracted_state.unsqueeze(0)
        return extracted_state.cuda()
    elif env == "color":
        # [agent, fish, green, red, r, X, Y]
        extracted_state = torch.zeros((3, 6))
        for i, state in enumerate(states):
            if i == 0:
                extracted_state[i, 0] = 1  # agent
                extracted_state[i, 4] = states[i, 0]  # X
                extracted_state[i, 5] = states[i, 1]  # Y
            else:
                extracted_state[i, 1] = 1  # fish
                if states[i, 2] == 1:
                    extracted_state[i, 2] = 1  # green
                else:
                    extracted_state[i, 3] = 1  # red
                extracted_state[i, 4] = states[i, 0]  # X
                extracted_state[i, 5] = states[i, 1]  # Y

        extracted_state = extracted_state.unsqueeze(0)
        return extracted_state.cuda()


def simplify_action(action):
    """[
        ("LEFT", "DOWN"),
        ("LEFT",),
        ("LEFT", "UP"),
        ("DOWN",),
        (),
        ("UP",),
        ("RIGHT", "DOWN"),
        ("RIGHT",),
        ("RIGHT", "UP")
    ]"""
    #              [0, 1, 2, 3, 4]
    action_space = [1, 3, 4, 5, 7]
    # ac_index = action[0].astype(int)
    # action = action_space[ac_index]
    action = action_space[action]
    return np.array([action])


def print_explaining(actions, prednames):
    action = torch.argmax(actions)
    return print(prednames[action])


def num_action_bf(action, trained=False):
    """
    prednames:  [
                'up_to_eat',
                'left_to_eat',
                'down_to_eat',
                'right_to_eat',
                'up_to_dodge',
                'down_to_dodge',
                'up_redundant',
                'down_redundant'
                'left_redundant',
                'right_redundant',
                'idle_redundant'
                ]

    env_actions
                [
                    ("LEFT",),
                    ("DOWN",),
                    (),
                    ("UP",),
                    ("RIGHT",),
                ]
    action_space = [1, 3, 4, 5, 7]
    """
    # if trained == True:
    #     action = torch.argmax(action)
    # up
    if action in [0, 4, 6]:
        return np.array([5])
    # left
    elif action in [1, 8]:
        return np.array([1])
    # down
    elif action in [2, 5, 7]:
        return np.array([3])
    # right
    elif action in [3, 9]:
        return np.array([7])
    # idle
    else:
        return np.array([4])


def extract_state_bf(obs, args):
    """
        states = [x, y, radius]
        extracted_states = [agent, fish, r, x, y]
        size: 3*5 3 fish, 5 features
        """
    states = torch.from_numpy(obs).squeeze()
    if args.alg == 'logic':
        if args.env == "bigfishm":
            # [agent, fish, r, X, Y]
            extracted_state = torch.zeros((3, 5))
            for i, state in enumerate(states):
                if i == 0:
                    extracted_state[i, 0] = 1  # agent
                    extracted_state[i, 2] = states[i, 2]  # radius
                    extracted_state[i, 3] = states[i, 0]  # X
                    extracted_state[i, 4] = states[i, 1]  # Y
                else:
                    extracted_state[i, 1] = 1  # fish
                    extracted_state[i, 2] = states[i, 2]  # radius
                    extracted_state[i, 3] = states[i, 0]  # X
                    extracted_state[i, 4] = states[i, 1]  # Y

            extracted_state = extracted_state.unsqueeze(0)
            return extracted_state.cuda()
        elif args.env == "bigfishc":
            # [agent, fish, green, red, X, Y]
            extracted_state = torch.zeros((3, 6))
            for i, state in enumerate(states):
                if i == 0:
                    extracted_state[i, 0] = 1  # agent
                    extracted_state[i, 4] = states[i, 0]  # X
                    extracted_state[i, 5] = states[i, 1]  # Y
                else:
                    extracted_state[i, 1] = 1  # fish
                    if states[i, 2] == 1:
                        extracted_state[i, 2] = 1  # green
                    else:
                        extracted_state[i, 3] = 1  # red
                    extracted_state[i, 4] = states[i, 0]  # X
                    extracted_state[i, 5] = states[i, 1]  # Y

            extracted_state = extracted_state.unsqueeze(0)
            return extracted_state.cuda()

    elif args.alg == 'ppo':
        state = obs.reshape(-1)
        # return torch.tensor(state, device='cuda:0')
        state = state.tolist()
        # if train:
        #     state = torch.nn.functional.normalize(torch.tensor(state), p=2, dim=0)
        #     state[2] = 0
        #     state[5] = -1  # small fish flag
        #     state[8] = 1  # big fish flag
        #     return state
        # else:
        return torch.tensor(state).cuda()
