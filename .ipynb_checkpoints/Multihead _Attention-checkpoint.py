################################################### MULTI-HEAD ATTENTION ##############################################################################

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


sequence_length = 4 # This is the maximum length of the input sequence(my name is Andrew)
batch_size = 1
input_dim  = 512  # this is the dimensional length of each input vector
d_model    = 512  # This is the dimensional length of the output vector from the attention head

x = torch.randn((batch_size, sequence_length, input_dim))  # this is a random input sequence. (This is being used coze we have not designed the position encoder yet.)

#print(x.size())

# create a query, key and value layer 
qkv_layer = nn.Linear(input_dim, 3*d_model) #we multiply by 3 so that we concatenate the 3 values of qkv
qkv = qkv_layer(x) # this is the concatenated value of query, key and value vector along the dimension axis (torch.Size([1, 4, 1536])) for each word

#print(qkv.size())

# just to check how all the values in the qkv vector are distributed.
y_val = torch.histc(qkv, bins=200, min=-3, max=3)
x_val = np.arange(-1, 1, 0.01) * 3
#plt.bar(x_val, y_val, align='center', color=['green'])
#plt.title('qkv distribution')


######################################### Attention head #####################################################################
num_heads = 8 # no of attention heads
head_dim = d_model // num_heads
qkv = qkv.reshape(batch_size, sequence_length, num_heads, 3 * head_dim)

#print('qkv:', qkv.size())
qkv = qkv.permute(0, 2, 1, 3) #(batch_size, num_heads, sequence_length, 3*head). we do this so that its easier to perform parallel operations on these last two dims
#print('qkv 1:', qkv.size())


# access the individual Q, K, V by splitting qkv into 3 chunks along the last dimension 
Q, K, V = torch.chunk(qkv, 3, dim=-1)

#print('Query:',Q.shape)
#print('Key:',K.shape)
#print('Value:',V.shape)

###################################################### SELF ATTENTION FOR MULTI-HEADS ###################################################
###################################################### IN PYTORCH ######################################################################
d_k = Q.size()[-1]

scaled = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
#print('scaled:', scaled.shape)

############################################## MASK #############################################
mask = torch.full(scaled.size(), float('-inf')) # we use -inf coz when we apply softmax, the exp(-inf)= 0 and this helps in the model not to cheat by looking in the future
mask = torch.triu(mask, diagonal=1)
#print(mask[0][1]) # this is a mask for input to a single head
#print((scaled + mask)[0][0])

scaled += mask
attention = F.softmax(scaled, dim=-1)
#print('attention of the first head:\n',attention[0][0])

values = torch.matmul(attention, V) # as per the eqtn of for self attention, to get the new values of a token, multiply te attention with the initial token value
#print('value:', values.shape) # this is more context aware than the initial value
#print('initial value:', V.shape)


###################### General function for performing self attention for either the encoder or decorder ##################################

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)

    return values, attention

## to test our code 
values, attention = scaled_dot_product(Q, K, V, mask=None)
#print('attention shape:', attention.shape)
#print('value shape:', values.shape) # for every single word
#print('encoder attention:\n', attention[0][0]) # attention for the first head

values, attention = scaled_dot_product(Q, K, V, mask=mask)
#print('decoder attention:\n', attention[0][0]) # attention for the first head

# concatenate all the values from the 8 headds into 1 output vector of ([1, 4, 512]) dim
values = values.reshape(batch_size, sequence_length, num_heads*head_dim)
#print('concat_value:', values.shape)

######################### FEED FORWARD PASS ####################################
linear_layer = nn.Linear(d_model, d_model)
out = linear_layer(values) # this gives a rich output vector with more context aware
#print('output:\n', out.shape)

################################################################################################################################################################
######################################################## FULL CODE #############################################################################################
# ##############################################################################################################################################################
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)

    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
        self.Linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size()
        print(f"x.size(): {x.size()}")
        qkv = self.qkv_layer(x)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3)
        print(f"qkv.size(): {qkv.size()}")
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        print(f"q size(): {q.size()}, k size(): {k.size()}, v size(): {v.size()},")
        values, attention = scaled_dot_product(q, k, v, mask)
        print(f"values.size(): {values.size()}, attention.size(): {attention.size()} ")
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        print(f"value.size(): {values.size()}")
        out = self.Linear_layer(values)
        print(f"out.size(): {out.size()}")

        return out

##################################################################################################################################################################
###################################################### Trial output ##############################################################################################
input_dim = 1024
d_model  = 512
num_heads = 8

batch_size = 30
sequence_length = 5

x = torch.randn((batch_size, sequence_length, input_dim))

model = MultiheadAttention(input_dim, d_model, num_heads)

out = model.forward(x)