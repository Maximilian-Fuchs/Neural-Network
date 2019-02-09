# ToDo: Add comments

import numpy as np


class Graph:
    """
    This class represents a Neural Network.
    You can append layers and units.
    Layers and Units are functions.
    The Graph can train and classify.
    """

    def __init__(self):
        self.graph = []

    def append_units(self, units):
        if len(self.graph) > 0:
            self.graph[-1].next_level = units
            units.prev_level = self.graph[-1]
        self.graph.append(units)

    def append_layer(self, layer):
        # register new level to the next
        self.graph[-1].next_level = layer
        layer.prev_level = self.graph[-1]
        self.graph.append(layer)

    def forward(self):
        level_result = []
        for level in self.graph:
            level_result = level.forward(level_result)
        y_guess = level_result
        return y_guess

    def backward(self):
        level_derivative_L = np.array([])
        for level in reversed(self.graph):
            level_derivative_L = level.backward(level_derivative_L)

    def update(self):
        for level in self.graph:
            level.update()


class InputUnitLevel:
    """
    This represents an InputUnit.
    """

    def __init__(self, x):
        self.next_level = None
        self.x = np.reshape(x, (1, x.shape[1]))

    def forward(self, _):
        return self.x

    def backward(self, _):
        pass

    def update(self):
        pass


class SigmoidUnitLevel():
    def __init__(self):
        self.next_level = None
        self.prev_level = None
        self.local_grad = None
        self.global_grad = None

    def forward(self, x):
        self.pre_activation = x
        self.activation = 1 / (1 + np.exp(-x))
        return self.activation

    def backward(self, grad_nl_L):
        """
        @param: grad_nl_L: gradient of next level with respect to Loss.
        """

        # calculate local gradient
        # ToDo:check if this is in fact the local gradient
        sig_x = self.activation
        self.local_grad = sig_x * (1 - sig_x)

        # if next level is a loss level:
        # this levels gradient is the loss levels gradient
        if isinstance(self.next_level, SquareLossUnitLevel):
            return grad_nl_L

        a = self.next_level.grad_nl_L
        b = self.next_level.next_level.local_grad
        c = self.next_level.W
        self.global_grad = np.dot(a, (b * c).T)
        # print(self.global_grad)
        return self.global_grad

    def update(self):
        pass


class SquareLossUnitLevel():
    def __init__(self, y_true):
        self.next_level = None
        self.prev_level = None
        self.y_true = y_true

    def forward(self, x):
        self.x = x
        return np.square(self.y_true - x)

    def backward(self, _):
        grad_L = 2 * (self.x - self.y_true)
        return grad_L

    def update(self):
        pass


class FullyConnectedLayer():
    """
    This class represents a fully connected Layer
    """

    def __init__(self, W, learning_rate=1):
        self.next_level = None
        self.prev_level = None
        # numpyfy if not already
        self.W = np.array(W)
        self.learning_rate = learning_rate

    def forward(self, input_):
        # numpyfy if not already
        input_ = np.array(input_)
        self.input_ = input_
        # return dot product
        return np.dot(input_, self.W)

    def backward(self, grad_nl_L):
        # see below and hex03 for proof
        self.grad_nl_L = grad_nl_L
        self.global_grad = grad_nl_L * self.next_level.local_grad * self.input_.T
        return self.global_grad

    def update(self):
        self.W = self.W - self.global_grad * self.learning_rate


################################
#            TESTS 1           #
################################

W = [[-1.4, -0.81],
     [-2.2, -1.7],
     [-0.27, -0.73]]
W = np.array(W)


p = [[0.75, 0.35, 0.46]]
p = np.array(p)

t = [[0, 1]]
t = np.array(t)

graph = Graph()
graph.append_units(InputUnitLevel(p))
graph.append_layer(FullyConnectedLayer(W))
graph.append_units(SigmoidUnitLevel())
graph.append_units(SquareLossUnitLevel(t))

loss = graph.forward()
graph.backward()
graph.update()
print(loss)




################################
#            TESTS 2           #
################################


from sklearn.datasets.samples_generator import make_classification

# Generate some data
X, y_true = make_classification(n_samples=400)

graph = Graph()
graph.append_units(InputUnitLevel(X[0].reshape(1,20)))
W = np.linspace(0, 1, 20 * 5).reshape(20, 5)
layer_W = FullyConnectedLayer(W)
graph.append_layer(layer_W)
graph.append_units(SigmoidUnitLevel())
T = np.linspace(0, 1, 5 * 1).reshape(5, 1)
layer_T = FullyConnectedLayer(T)
graph.append_layer(layer_T)
graph.append_units(SigmoidUnitLevel())
graph.append_units(SquareLossUnitLevel(y_true[0]))
loss = graph.forward()
graph.backward()
# print(loss)

################################
#            TESTS 3           #
################################

graph = Graph()
x = [[0,1]]
x = np.array(x)
t = [[1,1]]
t = np.array(t)
U = [[1.2, -1.2, -0.11],
     [0.3, 1.1, 0.65]]
U = np.array(U)

V = [[-0.25], [-1.1], [-0.09]]
V = np.array(V)
W = [[-2, 0.43]]
W = np.array(W)

graph.append_units(InputUnitLevel(x))
graph.append_layer(FullyConnectedLayer(U))
graph.append_units(SigmoidUnitLevel())
graph.append_layer(FullyConnectedLayer(V))
graph.append_units(SigmoidUnitLevel())
graph.append_layer(FullyConnectedLayer(W))
graph.append_units(SigmoidUnitLevel())
graph.append_units(SquareLossUnitLevel(t))

for i in range(1000000):
    loss = graph.forward()
    graph.backward()
    graph.update()

###########################
# Weight Derivatives test #
###########################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    sig_x = sigmoid(x)
    return sig_x * (1 - sig_x)


W = [[-1.4, -0.81],
     [-2.2, -1.7],
     [-0.27, -0.73]]
W = np.array(W)
p = np.array([[0.74, 0.35, 0.46]])  # input_
d_y = np.array([[1.75, -0.35]])  # grad_nl_L
print(np.dot(p, W))
d_sig_WP = d_sigmoid(np.array([[-1.96, -1.54]]))  # d_sigmoid(W @ p) #next_level.local_grad
# print(d_sig_WP)
# print(d_y * d_sig_WP * p.T) # grad_nl_L * next_level.local_grad * input_.T


####################
# Unit Derivatives #
####################
d_E_p = np.dot(d_y, (d_sig_WP * W).T)  # self.next_level.grad_nl_L @ (self.next_level.next_level.local_grad * self.next_level.W).T
print(d_E_p)

print(np.dot(np.arange(1, 3).reshape(1, 2) , np.arange(1, 3).reshape(2, 1)))

print((d_sig_WP * W).T)

print(type(SquareLossUnitLevel(4)))

print(np.array([[1, 2, 3, 4, 5]]).shape)
