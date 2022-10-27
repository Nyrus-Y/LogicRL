import random
import tqdm
import time
import argparse
import numpy as np
import torch
import os
from argparse import ArgumentParser
import pathlib

from src.coinjump.coinjump.imageviewer import ImageViewer
from src.coinjump.coinjump.coinjump.coinjump import CoinJump
from src.coinjump.coinjump.coinjump.paramLevelGenerator import ParameterizedLevelGenerator
from src.coinjump.coinjump.coinjump.paramLevelGenerator_keydoor import ParameterizedLevelGenerator_KeyDoor
from src.coinjump.coinjump.coinjump.paramLevelGenerator_dodge import ParameterizedLevelGenerator_Dodge
from src.coinjump.coinjump.coinjump.paramLevelGenerator_V1 import ParameterizedLevelGenerator_V1
from src.coinjump.coinjump_learn.training.data_transform import extract_state, sample_to_model_input_V1, collate
from src.coinjump.coinjump_learn.training.ppo_coinjump import ActorCritic
from src.coinjump.coinjump.coinjump.actions import coin_jump_actions_from_unified

from src.util import extract_for_cgen_explaining
from src.nsfr_utils import get_nsfr_model, update_initial_clauses, get_prob, get_data_loader
from src.logic_utils import get_lang, get_searched_clauses
from src.mode_declaration import get_mode_declarations
from src.clause_generator import ClauseGenerator
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, recall_score, roc_curve

KEY_r = 114
device = torch.device('cuda:0')


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.logic_states = []
        self.neural_states =[]
        self.action_probs = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.predictions = []

    def clear(self):
        del self.actions[:]
        del self.logic_states[:]
        del self.neural_states[:]
        del self.action_probs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.predictions[:]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int, default=1, help="Batch size in beam search")
    parser.add_argument("--e", type=int, default=6,
                        help="The maximum number of objects in one image")
    parser.add_argument("--dataset", choices=["coinjump_5a"],
                        help="Use coinjump dataset")
    parser.add_argument("--dataset-type", default="coinjump", help="coinjump env")
    parser.add_argument('--device', default='cuda:0',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument("--small-data", action="store_true", help="Use small training data.")
    parser.add_argument("--no-xil", action="store_true", help="Do not use confounding labels for clevr-hans.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--plot", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--t-beam", type=int, default=2, help="Number of rule expantion of clause generation.")
    parser.add_argument("--n-beam", type=int, default=8, help="The size of the beam.")
    parser.add_argument("--n-max", type=int, default=50, help="The maximum number of clauses.")
    parser.add_argument("--m", type=int, default=1, help="The size of the logic program.")
    parser.add_argument("--n-obj", type=int, default=2, help="The number of objects to be focused.")
    parser.add_argument("--epochs", type=int, default=101, help="The number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-2, help="The learning rate.")
    parser.add_argument("--n-data", type=float, default=200, help="The number of data to be used.")
    parser.add_argument("--pre-searched", action="store_true", help="Using pre searched clauses.")
    args = parser.parse_args()
    return args


def train_nsfr(args, NSFR, optimizer, train_loader, val_loader, test_loader, device, writer):
    bce = torch.nn.BCELoss()
    loss_list = []
    for epoch in range(args.epochs):
        loss_i = 0
        for i, sample in tqdm(enumerate(train_loader, start=0)):
            # to cuda
            imgs, target_set = map(lambda x: x.to(device), sample)

            # infer and predict the target probability
            V_T = NSFR(imgs)
            ##NSFR.print_valuation_batch(V_T)
            predicted = get_prob(V_T, NSFR, args)
            loss = bce(predicted, target_set)
            loss_i += loss.item()
            loss.backward()
            optimizer.step()

            # if i % 20 == 0:
            #    NSFR.print_valuation_batch(V_T)
            #    print("predicted: ", np.round(predicted.detach().cpu().numpy(), 2))
            #    print("target: ", target_set.detach().cpu().numpy())
            #    NSFR.print_program()
            #    print("loss: ", loss.item())

            # print("Predicting on validation data set...")
            # acc_val, rec_val, th_val = predict(
            #    NSFR, val_loader, args, device, writer, th=0.33, split='val')
            # print("val acc: ", acc_val, "threashold: ", th_val, "recall: ", rec_val)
        loss_list.append(loss_i)
        # rtpt.step(subtitle=f"loss={loss_i:2.2f}")
        writer.add_scalar("metric/train_loss", loss_i, global_step=epoch)
        print("loss: ", loss_i)
        # NSFR.print_program()
        if epoch % 20 == 0:
            NSFR.print_program()
            print("Predicting on validation data set...")
            acc_val, rec_val, th_val = predict(NSFR, val_loader, args, device, th=0.33, split='val')
            writer.add_scalar("metric/val_acc", acc_val, global_step=epoch)
            print("acc_val: ", acc_val)

            print("Predicting on training data set...")
            acc, rec, th = predict(NSFR, train_loader, args, device, th=th_val, split='train')
            writer.add_scalar("metric/train_acc", acc, global_step=epoch)
            print("acc_train: ", acc)

            print("Predicting on test data set...")
            acc, rec, th = predict(NSFR, test_loader, args, device, th=th_val, split='train')
            writer.add_scalar("metric/test_acc", acc, global_step=epoch)
            print("acc_test: ", acc)

    return loss


def predict(NSFR, loader, args, device, th=None, split='train'):
    predicted_list = []
    target_list = []
    count = 0
    ###NSFR = discretise_NSFR(NSFR, args, device)
    # NSFR.print_program()

    for i, sample in tqdm(enumerate(loader, start=0)):
        # to cuda
        imgs, target_set = map(lambda x: x.to(device), sample)

        # infer and predict the target probability
        V_T = NSFR(imgs)
        predicted = get_prob(V_T, NSFR, args)
        predicted_list.append(predicted.detach())
        target_list.append(target_set.detach())
        # if args.plot:
        #     imgs = to_plot_images_kandinsky(imgs)
        #     captions = generate_captions(
        #         V_T, NSFR.atoms, NSFR.pm.e, th=0.3)
        #     save_images_with_captions(
        #         imgs, captions, folder='result/kandinsky/' + args.dataset + '/' + split + '/', img_id_start=count, dataset=args.dataset)
        count += V_T.size(0)  # batch size

    predicted = torch.cat(predicted_list, dim=0).detach().cpu().numpy()
    target_set = torch.cat(target_list, dim=0).to(
        torch.int64).detach().cpu().numpy()

    if th == None:
        fpr, tpr, thresholds = roc_curve(target_set, predicted, pos_label=1)
        accuracy_scores = []
        print('ths', thresholds)
        for thresh in thresholds:
            accuracy_scores.append(accuracy_score(
                target_set, [m > thresh for m in predicted]))

        accuracies = np.array(accuracy_scores)
        max_accuracy = accuracies.max()
        max_accuracy_threshold = thresholds[accuracies.argmax()]
        rec_score = recall_score(
            target_set, [m > thresh for m in predicted], average=None)

        print('target_set: ', target_set, target_set.shape)
        print('predicted: ', predicted, predicted.shape)
        print('accuracy: ', max_accuracy)
        print('threshold: ', max_accuracy_threshold)
        print('recall: ', rec_score)

        return max_accuracy, rec_score, max_accuracy_threshold
    else:
        accuracy = accuracy_score(target_set, [m > th for m in predicted])
        rec_score = recall_score(
            target_set, [m > th for m in predicted], average=None)
        return accuracy, rec_score, th


def setup_image_viewer(coinjump):
    viewer = ImageViewer(
        "coinjump1",
        coinjump.camera.height,
        coinjump.camera.width,
        monitor_keyboard=True,
        # relevant_keys=set('W','A','S','D','SPACE')
    )
    return viewer


def create_coinjump_instance(seed=None, Dodge_model=False, Key_Door_model=False, V1=False):
    seed = random.randint(0, 100000000) if seed is None else seed

    # level_generator = DummyGenerator()
    if Dodge_model:
        coin_jump = CoinJump(start_on_first_action=True, Dodge_model=True)
        level_generator = ParameterizedLevelGenerator_Dodge()
    elif Key_Door_model:
        coin_jump = CoinJump(start_on_first_action=True, Key_Door_model=True)
        level_generator = ParameterizedLevelGenerator_KeyDoor()
    elif V1:
        coin_jump = CoinJump(start_on_first_action=True, V1=True)
        level_generator = ParameterizedLevelGenerator_V1()
    else:
        coin_jump = CoinJump(start_on_first_action=True)
        level_generator = ParameterizedLevelGenerator()

    level_generator.generate(coin_jump, seed=seed)
    coin_jump.render()

    return coin_jump


def parse_args():
    parser = ArgumentParser("Loads a model and lets it play coinjump")
    parser.add_argument("-m", "--model_file", dest="model_file", default=None)
    parser.add_argument("-s", "--seed", dest="seed", type=int)
    args = parser.parse_args()

    # TODO change path of model
    if args.model_file is None:
        # read filename from stdin
        current_path = os.path.dirname(__file__)
        model_name = input('Enter file name: ')
        model_file = os.path.join(current_path, 'ppo_coinjump_model', model_name)
        # model_file = f"../src/ppo_coinjump_model/{input('Enter file name: ')}"

    else:
        model_file = pathlib.Path(args.model_file)

    return args, model_file


def load_model(model_path, set_eval=True):
    with open(model_path, "rb") as f:
        model = torch.load(f)

    if isinstance(model, ActorCritic):
        model = model.actor
        model.as_dict = True

    if set_eval:
        model = model.eval()

    return model


def run():
    args, model_file = parse_args()

    model = load_model(model_file)

    seed = random.seed() if args.seed is None else int(args.seed)

    coin_jump = create_coinjump_instance(seed=seed, V1=True)
    #viewer = setup_image_viewer(coin_jump)

    # frame rate limiting
    fps = 10
    target_frame_duration = 1 / fps
    last_frame_time = 0

    args = get_args()
    buffer = RolloutBuffer()
    # nsfr = get_nsfr(env_name)

    # collect data
    max_states = 10000
    max_states = 100
    save_frequence = 1
    print_frequence = 1000
    step = 0
    collected_states = 0

    while collected_states < max_states:
        # # control framerate
        # current_frame_time = time.time()
        # # limit frame rate
        # if last_frame_time + target_frame_duration > current_frame_time:
        #     sl = (last_frame_time + target_frame_duration) - current_frame_time
        #     time.sleep(sl)
        #     continue
        # last_frame_time = current_frame_time  # save frame start time for next iteration

        # step game
        action = []
        step += 1

        if not coin_jump.level.terminated:
            model_input = sample_to_model_input_V1((extract_state(coin_jump), []))
            model_input = collate([model_input])
            prediction = model(model_input['state'])
            # 1 left 2 right 3 up
            action = coin_jump_actions_from_unified(torch.argmax(prediction).cpu().item() + 1)

            logic_state = extract_for_cgen_explaining(coin_jump)
            if step % save_frequence == 0:
                collected_states += 1
                buffer.logic_states.append(logic_state.detach())
                buffer.actions.append(torch.argmax(prediction.detach()))
                buffer.action_probs.append(prediction.detach())
                buffer.neural_states.append(model_input['state'])

                if collected_states % print_frequence == 0:
                    print("states collected: {} , max states: {}".format(collected_states, max_states))
        else:
            coin_jump = create_coinjump_instance(seed=seed, V1=True)
            action = []



        reward = coin_jump.step(action)
        # score = coin_jump.get_score()
        # np_img = np.asarray(coin_jump.camera.screen)
        # viewer.show(np_img[:, :, :3])

        # terminated = coin_jump.level.terminated
        # if viewer.is_escape_pressed:
        #     break

    print('data collected')

    env_name = 'coinjump_bm'
    current_path = os.getcwd()
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = os.path.join(current_path, 'data/lang/')

    writer = SummaryWriter(f"runs/{env_name}", purge_step=0)

    device = torch.device('cuda:0')
    # lang, clauses, bk, atoms = get_lang(
    #     lark_path, lang_base_path, args.dataset_type, args.dataset)
    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, 'coinjump', env_name)
    # clauses = update_initial_clauses(clauses, args.n_obj)
    bk_clauses = []
    # print("clauses: ", clauses)

    # Neuro-Symbolic Forward Reasoner for clause generation
    NSFR_cgen = get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device=device)  # torch.device('cpu'))
    mode_declarations = get_mode_declarations(args, lang, args.n_obj)
    # print("Maze terminated")

    cgen = ClauseGenerator(args, NSFR_cgen, lang, atoms,buffer, mode_declarations, device=device,
                           no_xil=args.no_xil)  # torch.device('cpu'))
    # generate clauses
    # if args.pre_searched:
    #     clauses = get_searched_clauses(lark_path, lang_base_path, args.dataset_type, args.dataset)
    # else:
    clauses = cgen.generate(clauses, T_beam=args.t_beam, N_beam=args.n_beam, N_max=args.n_max)
    print("====== ", len(clauses), " clauses are generated!! ======")

    # update
    # NSFR = get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device, train=True)
    # params = NSFR.get_params()
    # optimizer = torch.optim.RMSprop(params, lr=args.lr)
    # # ##optimizer = torch.optim.Adam(params, lr=args.lr)
    # train_loader, val_loader, test_loader = get_data_loader(buffer, args)
    # #
    # loss_list = train_nsfr(args, NSFR, optimizer, train_loader, val_loader, test_loader, device, writer)
    # #
    # # validation split
    # print("Predicting on validation data set...")
    # acc_val, rec_val, th_val = predict(
    #     NSFR, val_loader, args, device, th=0.33, split='val')
    #
    # print("Predicting on training data set...")
    # # training split
    # acc, rec, th = predict(
    #     NSFR, train_loader, args, device, th=th_val, split='train')
    #
    # print("Predicting on test data set...")
    # # test split
    # acc_test, rec_test, th_test = predict(
    #     NSFR, test_loader, args, device, th=th_val, split='test')
    #
    # print("training acc: ", acc, "threashold: ", th, "recall: ", rec)
    # print("val acc: ", acc_val, "threashold: ", th_val, "recall: ", rec_val)
    # print("test acc: ", acc_test, "threashold: ", th_test, "recall: ", rec_test)


if __name__ == "__main__":
    run()