# SimpleRNN in numpy
import numpy as np

timesteps = 100
input_features = 32
output_features = 64 # h (state) is the same as output

inputs = np.random.random((timesteps, input_features))

state_t = np.zeros(shape=(output_features,)) # init state

W = np.random.random((output_features, input_features))  # 64 x 32, W * X
U = np.random.random((output_features, output_features))  # 64 x 64, U * State(t-1)
b = np.random.random(output_features)  # 64 x 1

#    32          64
#  W * X + U * State(t-1) + b
#  Param Matrix: 64 row x 96 col  + 64 b
# tanh(64x96 x 96x1 + 64x1) = 64x1

successive_outputs = []
# this is only forward prop, w and u are never updated
for input_t in inputs:

    output_t = np.tanh( np.dot(W, input_t) + np.dot(U, state_t) + b) #input_t state_t => output_t
          #                 64x32 x 32x1 +  64x64 x 64x1  + 64

    successive_outputs.append(output_t)

    state_t = output_t  # update state_t using output_t

final_outputs = np.concatenate(successive_outputs, axis=0) #get the final_output with shape=(timesteps, output_features)
print(final_outputs)