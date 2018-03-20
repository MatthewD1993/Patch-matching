from torch2pytorch.read_lua_file import load_lua
from collections import OrderedDict
import torch

f = load_lua('Sintel128.7z')

# Keys extracted from state dict from saved checkpoint.
keys = ['feature_extractor_f.0.weight', 'feature_extractor_f.0.bias', 'feature_extractor_f.3.weight',
        'feature_extractor_f.3.bias', 'feature_extractor_f.5.weight', 'feature_extractor_f.5.bias',
        'feature_extractor_f.8.weight', 'feature_extractor_f.8.bias', 'feature_extractor_f.10.weight',
        'feature_extractor_f.10.bias', 'feature_extractor_f.12.weight', 'feature_extractor_f.12.bias',
        # Missed last conv layer.
        'feature_extractor_f.14.weight', 'feature_extractor_f.14.bias',]

l = [ (keys[i], f[i]) for i in range(len(keys))]

final = OrderedDict(l)
torch.save(final, 'bailer_weight_Sintel.ckpt')

# require 'cunn'
# require 'cudnn'
# b=a:parameters()[1]
# torch.save('Sintel.7z', b:float())