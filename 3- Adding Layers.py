import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)

>>>
array([[ 0.5031  -1.04185  -2.03875],
       [ 0.2434  -2.7332   -5.7633 ],
       [-0.99314  1.41254  -0.35655]])





# Training Data
# The nnfs package contains functions that we can use to create data. For example:
from nnfs.datasets import spiral_data

# For observations:
import matplotlib.pyplot as plt

X, y = spiral_data(samples=100, classes=3)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

# Adding color to the chart makes this more clear:
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.show()
# Spiral dataset colored by class above.





# Dense Layer Class
class Layer_Dense:

       # Layer initialization
       def __init__(self, n_inputs, n_neurons):
              # Initialize weights and biases
              pass # using pass statement as a placeholder

       # Forward pass
       def forward(self, inputs):
              # Calculate output values from inputs, weights and biases
              pass # using pass statement as a placeholder

# To continue the Layer_Dense class' code
# let's add the random initialization of weights and biases: 
# Layer Initialization
def __init__(self, n_inputs, n_neurons):
       self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
       self.biases = np.zeros((1, n_neurons))
 
# This example function call has returned a 2x5 array with data
# randomly sampled from a Gaussian distribution with a mean of 0.
import numpy as np
import nnfs
nnfs.init()
print(np.random.randn(2, 5))
>>>
[[ 1.7640524 0.4001572 0.978738 2.2408931 1.867558 ]
 [-0.9772779 0.95008844 -0.1513572 -0.10321885 0.41059852]]

# Next, the np.zeros function takes a desired array shape as 
# an argument and returns an array of that shape filled with zeros.
print(np.zeros((2, 5)))
>>>
[[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]

# To see an example of how our method initializes weights and biases:
import numpy as np
import nnfs

nnfs.init()

n_inputs = 2
n_neurons = 4

weights = 0.01 * np.random.randn(n_inputs, n_neurons)
biases = np.zeros((1, n_neurons))

print(weights)
print(biases)

>>>
[[ 0.01764052 0.00400157 0.00978738 0.02240893]
 [ 0.01867558 -0.00977278 0.00950088 -0.00151357]]
[[0. 0. 0. 0.]]

#On to our forward method — we need to update it with the dot product+biases calculation:
def forward(self, inputs):
       self.output = np.dot(inputs, self.weights) + self.biases

# Nothing new here, just turning the previous code into a method.
# Our full Layer_Dense class so far:
# Dense layer
class Layer_Dense:
 
       # Layer initialization
       def __init__(self, n_inputs, n_neurons):
              self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
              self.biases = np.zeros((1, n_neurons))
 
       # Forward pass
       def forward(self, inputs):
              self.output = np.dot(inputs, self.weights) + self.biases

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Let's see output of the first few samples:
print(dense1.output[:5])



# Go ahead and run everything. 
# Full code up to this point:
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense layer
class Layer_Dense:

       # Layer initialization
       def __init__(self, n_inputs, n_neurons):
              # Initialize weights and biases
              self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
              self.biases = np.zeros((1, n_neurons))

       # Forward pass
       def forward(self, inputs):
              # Calculate output values from inputs, weights and biases
              self.output = np.dot(inputs, self.weights) + self.biases

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Let's see output of the first few samples:
print(dense1.output[:5])

>>>
[[ 0.0000000e+00 0.0000000e+00 0.0000000e+00]
 [-1.0475188e-04 1.1395361e-04 -4.7983500e-05]
 [-2.7414842e-04 3.1729150e-04 -8.6921798e-05]
 [-4.2188365e-04 5.2666257e-04 -5.5912682e-05]
 [-5.7707680e-04 7.1401405e-04 -8.9430439e-05]]
