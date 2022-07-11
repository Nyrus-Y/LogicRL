import sys

sys.path.append('src/')
from src.logic_utils import get_lang

import torch
import numpy as np

lark_path = 'src/lark/exp.lark'
lang_base_path = 'data/lang/'
lang, clauses, bk, atoms = get_lang(
    lark_path, lang_base_path, 'coinjump1', 'coinjump1')

from percept import SlotAttentionPerceptionModule, YOLOPerceptionModule
from facts_converter import FactsConverter
from nsfr import NSFReasoner
from logic_utils import build_infer_module, generate_atoms
from valuation import RLValuationModule

device = torch.device('cpu')


def get_nsfr_model(lang, clauses, atoms, bk, device):
    PM = YOLOPerceptionModule(e=4, d=11, device=device)
    VM = RLValuationModule(
        lang=lang, device=device)
    FC = FactsConverter(lang=lang, perception_module=PM,
                        valuation_module=VM, device=device)
    IM = build_infer_module(clauses, atoms, lang,
                            m=len(clauses), infer_step=2, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(perception_module=PM, facts_converter=FC,
                       infer_module=IM, atoms=atoms, bk=bk, clauses=clauses)
    return NSFR


NSFR = get_nsfr_model(lang, clauses, atoms, bk, device=device)

metric = [[0, 0, 0, 1, 0.21, 0.2],
          [1, 0, 0, 0, 0.18, 0.22],
          [0, 0, 1, 0, 0.9, 0.8],
          [0, 1, 0, 0, 0.5, 0.5]]
x = torch.tensor(np.array(metric), dtype=torch.float32).unsqueeze(0)

V_T = NSFR(x)

NSFR.print_valuation_batch(V_T)
