# A single neuron:
inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

# Our output would be calculated up to this point like:
output = (inputs[0]*weights[0] +
          inputs[1]*weights[1] +
          inputs[2]*weights[2] + bias)

print(output)

>>>
2.3


# A Layer of Neurons
# 3 neurons in a layer and 4 inputs:
inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0,87]

bias1 = 2
bias2 = 3
bias3 = 0.5

outputs = [
        # Neuron 1:
        inputs[0]*weights1[0] +
        inputs[1]*weights1[1] +
        inputs[2]*weights1[2] +
        inputs[3]*weights1[3] + bias1,

        # Neuron 2:
        inputs[0]*weights2[0] + 
        inputs[1]*weights2[1] +
        inputs[2]*weights2[2] +
        inputs[3]*weights2[3] + bias2,

        # Neuron 3:
        inputs[0]*weights3[0] +
        inputs[1]*weights3[1] +
        inputs[2]*weights3[2] +
        inputs[3]*weights3[3] + bias3]

print(outputs)

[4.8, 1.21, 2.385]

# We changed the code to use loops instead of the hardcoded operations.
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# Output of current layer
layer_outputs = []
# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    # Zeroed output of given neuron
    neuron_output = 0
    # For each input and weight to the neuron
    for n_input, weight in zip(inputs, neuron_weights):
        # Multiply this input and weight to the neuron
        # and add to the neuron's output variable
        neuron_output += n_input*weight
    # Add bias
    neuron_output += neuron_bias
    # Put neuron's result to the layer's output list
    layer_outputs.append(neuron_output)

print(layer_outputs)

>>>
[4.8, 1.21, 2.385]




# Tensors, Arrays and Vectors
# This is an example of a simple list:
l = [1, 5, 6, 2]

# A list of lists:
lol = [[1, 5, 6, 2],
       [3, 2, 1, 3]]

# A list of lists of lists!
lolol = [[[1, 5, 6, 2],
          [3, 2, 1, 3]],
         [[5, 2, 1, 2],
          [6, 4, 8, 4]],
         [[2, 8, 5, 3],
          [1, 1, 9, 4]]]

# The below list of lists cannot be an array because it is not homologous.
another_list_of_lists = [[4, 2, 3],
                         [5, 1]]

# A matrix can be array but can't all arrays be matrices.
list_matrix_array = [[4,2],
                     [5,1],
                     [8,2]]

# With 3-dimensional arrays, like in lolol below, we'll have a 3rd level of brackets:
lolol = [[[1, 5, 6, 2],
          [3, 2, 1, 3]],
         [[5, 2, 1, 2],
          [6, 4, 8, 4]],
         [[2, 8, 5, 3],
          [1, 1, 9, 4]]]

# The first level of this array contains 3 matrices:
         [[1, 5, 6, 2],
          [3, 2, 1, 3]]

         [[5, 2, 1, 2],
          [6, 4, 8, 4]]

And

         [[2, 8, 5, 3],
          [1, 1, 9, 4]]
# This array's shape is (3, 2, 4) and type is 3D Array.
# An array as an ordered homologous container for numbers,
# and mostly use this term when working with the NumPy package 
# since that's what the main data structure is called within it.





# Dot Product and Vector Addition
a = [1 ,2 ,3]
b = [2, 3, 4]

# To obtain the dot product:
dot_product = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
print(dot_product)





# A Single Neuron with NumPy
import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

outputs = np.dot(weights, inputs) + bias
print(outputs)

>>>
4.8


# A Layer of Neurons with NumPy
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

layer_outputs = np.dot(weights, inputs) + biases

print(layer_outputs)

>>>
array([4.8  1.21  2.385])

# Code for the dot product applied to the layer of neurons.
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

outputs = np.dot(weights, inputs) + biases

np.dot(weights, inputs) = [np.dot(weights[0], inputs),
np.dot(weights[1], inputs), np.dot(weights[2], inputs)]
= [2.8, -1.79, 1.885]

# Code for the sum of the dot product and bias with a layer of neurons.
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

outputs = np.dot(weights, inputs) + biases
>>> array([4.8  1.21  2.385])





# Transposition for the Matrix Product
a = [1, 2, 3]
print(np.array([a]))

>>>
array([[1, 2, 3]])

# Where np.expand_dims() adds a new dimension at the index pf the axis. 
a = [1, 2, 3]
print(np.expand_dims(np.array(a), axis=0))

>>>
array([[1, 2, 3]])

# Matrix product with NumPy code:
import numpy as np

a = [1, 2, 3]
b = [2, 3, 4]

a = np.array([a])
b = np.array([b]).T

print(np.dot(a, b))

>>>
array([[20]])





# A Layer of Neurons & Batch of Data w/NumPy
import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

layer_outputs = np.dot(inputs, np.array(weights).T) + biases

print(layer_outputs)

>>>
array([[ 4.8   1.21   2.385],
       [ 8.9  -1.81   0.2  ],
       [ 1,41  1.051  0.026]])