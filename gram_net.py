import numpy as np
import cudarray as ca
import deeppy as dp
from deeppy.base import Model
from deeppy.parameter import Parameter
from style_network import Convolution
from style_network import gram_matrix


class GramNet(Model):
    def __init__(self, layers, style_weights):
        # Map weights (in convolution indices) to layer indices
        self.style_weights = np.zeros(len(layers))
        layers_len = 0
        conv_idx = 0
        for l, layer in enumerate(layers):
            if isinstance(layer, dp.Activation):
                self.style_weights[l] = style_weights[conv_idx]
                if style_weights[conv_idx] > 0:
                    layers_len = l+1
                conv_idx += 1

        # Discard unused layers
        layers = layers[:layers_len]

        # Wrap convolution layers for better performance
        self.layers = [Convolution(l) if isinstance(l, dp.Convolution) else l for l in layers]

    def compute_grams(self, image):
        # Setup network
        x_shape = image.shape
        self.x = Parameter(image)
        self.x._setup(x_shape)
        for layer in self.layers:
            layer._setup(x_shape)
            x_shape = layer.y_shape(x_shape)

        # Precompute subject features and style Gram matrices
        self.style_grams = [None]*len(self.layers)
        next_style = ca.array(image)
        for l, layer in enumerate(self.layers):
            next_style = layer.fprop(next_style)
            if self.style_weights[l] > 0:
                gram = gram_matrix(next_style)
                self.style_grams[l] = gram

        return self.style_grams;

    @property
    def image(self):
        return np.array(self.x.array)

    @property
    def _params(self):
        return [self.x]

    def _update(self):
        raise NotImplementedError("This network can only compute the stlye gram matrix.")
