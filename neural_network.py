import os
import keras.models
import keras.layers
import keras.utils
import matplotlib.pyplot as plt
import model
import numpy
import enum

class Optimizer(enum.Enum):
    Adadelta = "adadelta"
    Adagrad = "adagrad"
    Adam = "adam"
    Adamax = "adamax"
    Nadam = "nadam"
    RMSprop = "rmsprop"
    SGD = "sgd"
    Ftrl = "ftrl"
    LossScaleOptimizer = "lossscaleoptimizerv1"
    LossScaleOptimizerV3 = "lossscaleoptimizerv3"

class NeuralNetwork(model.Model):
    def __init__(self,
                 input_shape: list[int],
                 network_shape: list[int],
                 output_shape: list[int],
                 input_activation: str = 'tanh',
                 network_activation: str = 'tanh',
                 output_activation: str = 'tanh',
                 loss: str = 'binary_crossentropy',
                 optimizer: Optimizer = Optimizer.Adam,
                 metrics: list[str] = ['accuracy']) -> None:
        self.input_shape: list[int] = input_shape
        self.network_shape: list[int] = network_shape
        self.output_shape: list[int] = output_shape
        self.shape: str = "%sx%sx%s" % (input_shape[0], "x".join([str(_) for _ in network_shape]), output_shape[0])
        self.model = keras.models.Sequential()

        assert len(input_shape) > 0
        assert len(network_shape) > 0
        assert len(output_shape) > 0
        
        self.model.add(keras.layers.Dense(network_shape[0], input_shape=input_shape, activation=input_activation))
        for layer in network_shape[1:]:
            self.model.add(keras.layers.Dense(layer, activation=network_activation))
        self.model.add(keras.layers.Dense(output_shape[0], activation=output_activation))
        self.model.compile(loss=loss, optimizer=optimizer.value, metrics=metrics)

    def fit(self, trainset: dict) -> None:
        self.model.fit(trainset['x'], trainset['y'], epochs=10, batch_size=5, verbose=1)

    def predict(self, inputs: list) -> list:
        return self.model.predict(inputs, verbose=0)

    def evaluate(self, testset: dict) -> tuple:
        loss, acc = self.model.evaluate(testset['x'], testset['y'], verbose=0)
        return numpy.float64(loss),  numpy.float64(acc)

    def plot(self):
        return keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True, show_dtype=True, show_layer_activations=True)

    def dump(self, path: str) -> None:
        if not os.path.exists(path):
            os.mkdir(path)
        self.model.save_weights(os.path.join(path, "model.h5"))
        with open(os.path.join(path, "model.json"), "w") as json_file:
            json_file.write(self.model.to_json())
