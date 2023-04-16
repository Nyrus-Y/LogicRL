from .valuation_func_a import *


class AValuationModule(nn.Module):
    """A module to call valuation functions.
        Attrs:
            lang (language): The language.
            device (device): The device.
            layers (list(nn.Module)): The list of valuation functions.
            vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
            attrs (dic(term->tensor)): The dictionary that maps an attribute term to the corresponding one-hot encoding.
            dataset (str): The dataset.
    """

    def __init__(self, lang, device, pretrained=True):
        super().__init__()
        self.lang = lang
        self.device = device
        self.layers, self.vfs = self.init_valuation_functions(
            device, pretrained)

    def init_valuation_functions(self, device, pretrained):
        """
            Args:
                device (device): The device.
                pretrained (bool): The flag if the neural predicates are pretrained or not.

            Retunrs:
                layers (list(nn.Module)): The list of valuation functions.
                vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
        """
        layers = []
        vfs = {}  # pred name -> valuation function
        v_type = TypeValuationFunction()
        vfs['type'] = v_type

        # TODO
        v_closeby = ClosebyValuationFunction(device)
        vfs['closeby'] = v_closeby
        # vfs['closeby'].load_state_dict(torch.load(
        #     '../src/weights/neural_predicates/closeby_pretrain.pt', map_location=device))
        # vfs['closeby'].eval()
        layers.append(v_closeby)
        # print('Pretrained  neural predicate closeby have been loaded!')

        v_on_left = OnLeftValuationFunction()
        vfs['on_left'] = v_on_left
        layers.append(v_on_left)

        v_on_right = OnRightValuationFunction()
        vfs['on_right'] = v_on_right
        layers.append(v_on_right)

        v_same_row = SameRowValuationFunction()
        vfs['same_row'] = v_same_row
        layers.append(v_same_row)

        v_above_row = AboveRowValuationFunction()
        vfs['above_row'] = v_above_row
        layers.append(v_above_row)

        v_top5cars = Top5CarsValuationFunction()
        vfs['top5car'] = v_top5cars
        layers.append(v_top5cars)

        v_bottom5cars = Bottom5CarsValuationFunction()
        vfs['bottom5car'] = v_bottom5cars
        layers.append(v_bottom5cars)

        return nn.ModuleList([v_type, v_closeby, v_on_left, v_on_right, v_same_row, v_above_row, v_top5cars, v_bottom5cars]), vfs

    def forward(self, zs, atom):
        """Convert the object-centric representation to a valuation tensor.

            Args:
                zs (tensor): The object-centric representaion (the output of the YOLO model).
                atom (atom): The target atom to compute its proability.

            Returns:
                A batch of the probabilities of the target atom.
        """
        # term: logical term
        # arg: vector representation of the term
        # zs = self.preprocess(zs)
        args = [self.ground_to_tensor(term, zs) for term in atom.terms]
        # call valuation function
        return self.vfs[atom.pred.name](*args)

    def ground_to_tensor(self, term, zs):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.
        """
        term_index = self.lang.term_index(term)
        if term.dtype.name == 'object':
            return zs[:, term_index]
        elif term.dtype.name == 'image':
            return zs
        else:
            # other attributes
            return self.term_to_onehot(term, batch_size=zs.size(0))

    def term_to_onehot(self, term, batch_size):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.

            Return:
                The tensor representation of the input term.
        """
        if term.dtype.name == 'color':
            return self.to_onehot_batch(self.colors.index(term.name), len(self.colors), batch_size)
        elif term.dtype.name == 'shape':
            return self.to_onehot_batch(self.shapes.index(term.name), len(self.shapes), batch_size)
        elif term.dtype.name == 'material':
            return self.to_onehot_batch(self.materials.index(term.name), len(self.materials), batch_size)
        elif term.dtype.name == 'size':
            return self.to_onehot_batch(self.sizes.index(term.name), len(self.sizes), batch_size)
        elif term.dtype.name == 'side':
            return self.to_onehot_batch(self.sides.index(term.name), len(self.sides), batch_size)
        elif term.dtype.name == 'type':
            return self.to_onehot_batch(self.lang.term_index(term), len(self.lang.get_by_dtype_name(term.dtype.name)),
                                        batch_size)
        else:
            assert True, 'Invalid term: ' + str(term)

    def to_onehot_batch(self, i, length, batch_size):
        """Compute the one-hot encoding that is expanded to the batch size.
        """
        onehot = torch.zeros(batch_size, length, ).to(self.device)
        onehot[:, i] = 1.0
        return onehot