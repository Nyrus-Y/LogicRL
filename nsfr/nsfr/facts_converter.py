import torch
import torch.nn as nn
from tqdm import tqdm

from .fol.logic import NeuralPredicate


class FactsConverter(nn.Module):
    """
    FactsConverter converts the output fromt the perception module to the valuation vector.
    """

    def __init__(self, lang, valuation_module, device=None):
        super(FactsConverter, self).__init__()
        # self.e = perception_module.e
        self.e = 0
        #self.d = perception_module.d
        self.d =0
        self.lang = lang
        self.vm = valuation_module  # valuation functions
        self.device = device

    def __str__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.d)

    def __repr__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.d)

    def forward(self, Z, G, B):
        return self.convert(Z, G, B)

    def get_params(self):
        return self.vm.get_params()

    def init_valuation(self, n, batch_size):
        v = torch.zeros((batch_size, n)).to(self.device)
        v[:, 1] = 1.0
        return v

    def filter_by_datatype(self):
        pass

    def to_vec(self, term, zs):
        pass

    def __convert(self, Z, G):
        # Z: batched output
        vs = []
        for zs in tqdm(Z):
            vs.append(self.convert_i(zs, G))
        return torch.stack(vs)

    def convert(self, Z, G, B):
        batch_size = Z.size(0)

        # V = self.init_valuation(len(G), Z.size(0))
        V = torch.zeros((batch_size, len(G))).to(
            torch.float32).to(self.device)
        #dummy_zeros = nn.Parameter(torch.zeros((batch_size, len(G))).to(
        #    torch.float32).to(self.device))
        self.dummy_zeros = torch.zeros((batch_size, len(G)), requires_grad=True).to(
            torch.float32).to(self.device)
        self.dummy_zeros.requires_grad_()
        self.dummy_zeros.retain_grad()
        for i, atom in enumerate(G):
            if type(atom.pred) == NeuralPredicate and i > 1:
                V[:, i] = V[:, i] + self.vm(Z, atom)
            elif atom in B:
                # V[:, i] += 1.0
                V[:, i] = V[:, i] + torch.ones((batch_size,)).to(
                    torch.float32).to(self.device)
        V[:, 1] = torch.ones((batch_size,), requires_grad=True).to(
            torch.float32).to(self.device)
        V = V + self.dummy_zeros
        # V.retain_grad()
        return V

    def convert_i(self, zs, G):
        v = self.init_valuation(len(G))
        for i, atom in enumerate(G):
            if type(atom.pred) == NeuralPredicate and i > 1:
                v[i] = self.vm.eval(atom, zs)
        return v

    def call(self, pred):
        return pred
