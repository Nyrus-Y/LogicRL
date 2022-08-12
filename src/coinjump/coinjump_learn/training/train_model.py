import pickle
import random
from argparse import ArgumentParser
import pathlib
import numpy as np

import torch.optim
import torch.utils.data._utils.collate

from src.CoinJump.coinjump.coinjump import CJA_NUM_EXPLICIT_ACTIONS
from src.coinjump_learn.models.mlpController import MLPController
from src.coinjump_learn.training.data_transform import sample_to_model_input, for_each_tensor, collate


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="dataset")
    args = parser.parse_args()

    args.dataset_path = pathlib.Path(args.dataset)

    return args

def load_samples(path, train_test_split=0.80):
    with open(path, "rb") as f:
        dataset = pickle.load(f)

    train_test_split_abs = int(train_test_split * len(dataset))

    # flatten out steps
    samples_train = [sample_to_model_input(step) for recording in dataset[:train_test_split_abs] for step in recording[0]]
    samples_test = [sample_to_model_input(step) for recording in dataset[:train_test_split_abs] for step in recording[0]]

    return samples_train, samples_test


def extract_batch(o, idx, size):
    if isinstance(o, torch.Tensor):
        return o[idx:idx+size]
    if isinstance(o, list):
        return [extract_batch(e, idx, size) for e in o]
    if isinstance(o, dict):
        return {k: extract_batch(v, idx, size) for k,v in o.items()}
    raise ValueError("unexpected object type")



def shuffle_data(o):
    return for_each_tensor(o, lambda tensor: tensor[torch.randperm(tensor.shape[0])])


def count_actions(actions):
    counts = torch.zeros(CJA_NUM_EXPLICIT_ACTIONS).cuda()
    action_num, action_counts = torch.unique(actions, return_counts=True)
    for action, count in zip(action_num, action_counts):
        counts[action] = count
    return counts


def compute_action_weights(actions):
    counts = count_actions(actions)
    for i, c in enumerate(counts):
        if c == 0:
            counts[i] = 0.99
        else:
            counts[i] = 1.0 / c
    return counts


def equalize_samples(samples):
    action_samples = [[] for _ in range(CJA_NUM_EXPLICIT_ACTIONS)]
    for sample in samples:
        action_samples[sample['action']].append(sample)
    counts = np.array([len(s) for s in action_samples])
    print(f"counts: {counts}")
    non_zero_idx = np.where(counts != 0)[0] # np.nonzero(counts)
    median = int(np.median(counts[np.where(counts != 0)]))
    print(f"median # of samples: {median}")

    samples = []
    for i in non_zero_idx:
        if counts[i] >= median:
            samples.extend(action_samples[i][:median])
        else:
            multiplier = median // counts[i]
            samples.extend(action_samples[i] * multiplier)
    return samples


def run():
    args = parse_args()
    train_samples, test_samples = load_samples(args.dataset_path)
    #train_samples = equalize_samples(train_samples)
    train_samples = collate(train_samples)
    test_samples = collate(test_samples)
    num_train_samples = len(train_samples['state']['entities'])

    weights = compute_action_weights(train_samples['action'])

    #print("train sample 0:", train_samples[0])

    seed = 12345
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(0)

    num_epochs = 30
    batch_size = 128
    lr = 0.01

    model = MLPController().cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss = torch.nn.CrossEntropyLoss(weight=weights).cuda()
    #loss = torch.nn.CrossEntropyLoss().cuda()

    losses = []
    print(f"{num_epochs} epochs. batch size {batch_size}. {num_train_samples//batch_size} steps.")
    for epoch in range(num_epochs):
        print(f"epoch {epoch}")
        train_samples = shuffle_data(train_samples)

        for i in range(0, num_train_samples, batch_size):
            optimizer.zero_grad()

            batch = extract_batch(train_samples, i, batch_size)
            y = model(batch['state'])
            loss_value = loss(y, batch['action'])
            loss_value.backward()
            optimizer.step()

            if i % 100 == 0:
                loss_plain = loss_value.detach().cpu().item()
                print(f"step {i}; (loss {loss_plain})")
        scheduler.step()

    y_pred = torch.argmax(model(test_samples['state']), 1).detach().cpu()
    y_target = test_samples['action'].detach().cpu()
    pred_counts = count_actions(y_pred)
    target_counts = count_actions(y_target)
    print(f"stats pred: {pred_counts}")
    print(f"stats targ: {target_counts}")
    num_test_samples = len(test_samples['action'])
    correct = torch.sum(y_target == y_pred)
    print(f"accuracy {(correct/num_test_samples)*100:.3f}% ({correct}/{num_test_samples})")


    save_path = f"../models/mlpController_{num_epochs}_{lr}_{seed}.pkl"
    print(f"saving to {save_path}")
    torch.save(model, save_path)
    torch.save({
        "loss": losses
    }, f"../models/mlpController_{num_epochs}_{lr}_{seed}_meta.pkl")


if __name__ == "__main__":
    run()
