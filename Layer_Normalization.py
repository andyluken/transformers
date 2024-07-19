import torch
from torch import nn

class LayerNormalization():
    def __init__(self, parameters_shape, eps=1e-5):
        self.eps = eps
        self.parameters_shape = parameters_shape
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, input):
        dims = [-(i +1) for i in range(len(self.parameters_shape))]
        mean = input.mean(dim=dims, keepdim=True)
        print(f"Mean \n ({mean.size()}): \n {mean}")
        var = ((input - mean)**2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        print(f"Standard deviation \n ({std.size()}): \n {std}")
        y = (input - mean) / std
        print(f"y\n ({y.size()}) = \m {y}")
        out = self.gamma * y + self.beta
        print(f"out \n ({out.size()}) = \n {out}")
        return out
    

batch_size = 3
sentence_length = 5
embedding_dim = 8 
inputs = torch.randn(sentence_length, batch_size, embedding_dim)

#print(f"input \n ({inputs.size()}) = \n {inputs}")

layer_norm = LayerNormalization(inputs.size()[-1:])  
out = layer_norm.forward(inputs)    


print(out[0].mean(), out[0].std())

