import random
from argparse import ArgumentParser
import pathlib
import pickle
import sys

from src.CoinJump.coinjump.coinjump import ParameterizedLevelGenerator
from src.CoinJump.coinjump.coinjump import CoinJump
from src.CoinJump.coinjump.coinjump import unify_coin_jump_actions

#fix some imports for unpickling
from src import coinjump as cjActions, coinjump as cjEntityEncoding, coinjump as cjCoin
from src.coinjump_learn.training.data_transform import extract_state, state_to_extended_repr, replace_bools

sys.modules['coinjump1.actions'] = cjActions
sys.modules['coinjump1.entityEncoding'] = cjEntityEncoding
sys.modules['coinjump1.coin'] = cjCoin


def create_coinjump_instance(seed=None):
    seed = random.randint(0, 10000) if seed is None else seed

    coin_jump = CoinJump(start_on_first_action=True)
    level_generator = ParameterizedLevelGenerator()
    level_generator.generate(coin_jump, seed=seed)
    coin_jump.render()

    return coin_jump


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-r", "--recording_dir", dest="recording_dir")
    parser.add_argument("-s", "--save_file", dest="save_file")
    parser.add_argument("-sfi", "--save_file_info", dest="save_file_info", default=None)
    parser.add_argument("-f", "--format", dest="format", default="compact")
    args = parser.parse_args()

    recording_dir = pathlib.Path(args.recording_dir)

    return args, recording_dir

def run():
    args, recording_dir = parse_args()

    data_format = args.format
    if data_format == "actionpair":
        compact_format = False
    elif data_format == "compact":
        compact_format = True
    else:
        raise RuntimeError("invalid format")

    recordings = load_recordings(recording_dir)

    dataset = []

    for recording in recordings:
        data = recording[0]
        source_file_name = recording[1]
        actions = data["actions"]
        actions_count = len(actions)

        swap_coins = None  # keep track if we need to exchange the coins in position
        coin0_x = None

        collisions_count = 0
        collision_sequence = [-1, -1, -1, -1, -1]  # collisions so far. flag=0, pu=1, enemy=2, coin0=3, coin1=4
        backlog = 0

        coin_jump = create_coinjump_instance(data['meta']['level']['seed'])

        state_records = [None] * (actions_count+1)  # will contain (state, action) pairs

        # step through the game and record the last state with a NOOP action
        for step in range(actions_count + 1):
            action = actions[step] if step != actions_count else cjActions.CJA_NOOP

            state = extract_state(coin_jump)
            if compact_format:
                ext_repr, swap_coins, coin0_x = state_to_extended_repr(state, coin_jump, swap_coins=swap_coins, coin0_x=coin0_x)

                state_collisions = ext_repr[-5:]  # flag, PowerUp, enemy, coin0, coin1
                j = 0
                if any(state_collisions):
                    while any(state_collisions):
                        if state_collisions[j]:
                            collision_sequence[collisions_count] = j
                            collisions_count += 1

                            # set target for backlog steps
                            for k in range(backlog):
                                state_records[step-1-k][-5+j] = 1
                            state_collisions[j] = 0  # clear collision flag
                        j += 1
                    backlog = 0

                # next entity to interact/collide with. (Will be filled by backlog loop)
                # encoding like collisions: flag, PU, enemy, coin0, coin1
                target = [0, 0, 0, 0, 0]

                record = [*replace_bools(ext_repr), *collision_sequence, *target]
                backlog += 1
            else:
                record = (state, unify_coin_jump_actions(action))
            state_records[step] = record

            # step game
            coin_jump.step(action)

            if step == 0 or step == actions_count:
                print(record)

        #if not compact_format:
        state_records = [
            state_records,
            source_file_name
        ]
        dataset.append(state_records)

    save_file_path = pathlib.Path(args.save_file)
    if "default" in args.save_file_info:
        save_file_path = save_file_path.joinpath("coinjump1")
    if "format" in args.save_file_info:
        save_file_path = save_file_path.parent / f"{save_file_path.name}_{data_format}"
    if "count" in args.save_file_info:
        save_file_path = save_file_path.parent / f"{save_file_path.name}_{len(dataset)}"
    if "ext" in args.save_file_info:
        save_file_path = save_file_path.parent / f"{save_file_path.name}.pkl"

    with open(save_file_path, "wb") as f:
        pickle.dump(dataset, f)

    print("Conversion finished")


def load_recordings(recording_dir: pathlib.Path):
    recordings = []
    dir_warning_printed = False
    for path in recording_dir.iterdir():
        if path.is_dir():
            if not dir_warning_printed:
                print("warn: encountered subdirectory in recording dir")
                dir_warning_printed = True
            continue
        if path.is_file():
            recordings.append([load_recording(path), path])
    return recordings


def load_recording(replay_file):
    with open(replay_file, 'rb') as f:
        #data = {
        #    'actions': actions, =[ACTION,, ...]
        #    'meta': coinjump1.level.get_representation(),
        #    'score': coinjump1.score
        #}
        data = pickle.load(f)
        print("loading", data)
        return data


if __name__ == "__main__":
    run()
