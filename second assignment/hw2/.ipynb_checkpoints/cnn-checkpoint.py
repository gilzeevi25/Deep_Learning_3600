import torch
import torch.nn as nn
import itertools as it
from typing import Sequence

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======

        N = len(self.channels)
        P = self.pool_every
        conv_block = 0 #conv+activation layer block
        from math import floor
        pooling_layers = 0
        for i in range(N):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, self.channels[0], **self.conv_params)) #first input layer
            else:
                layers.append(nn.Conv2d(self.channels[i-1], self.channels[i], **self.conv_params))
            layers.append( ACTIVATIONS[self.activation_type]( **self.activation_params ) ) #activation layer

            conv_block += 1
            if conv_block % P == 0 & pooling_layers < floor(N/P): #Add pooling layer every P convolutional layers and up to floor(N/P)
                layers.append( POOLINGS[ self.pooling_type ]( **self.pooling_params ) )
                pooling_layers += 1

        seq = nn.Sequential(*layers)
        return seq


    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            from math import floor

            # to simplify, assume all are ints and not tuples (symmetric kernel, stride, ...)
            conv_kernel = self.conv_params['kernel_size'] if 'kernel_size' in self.conv_params.keys() else 1
            conv_stride = self.conv_params['stride'] if 'stride' in self.conv_params.keys() else 1
            conv_padding = self.conv_params['padding'] if 'padding' in self.conv_params.keys() else 0
            conv_dilation = self.conv_params['dilation'] if 'dilation' in self.conv_params.keys() else 1


            pool_kernel = self.pooling_params['kernel_size'] if 'kernel_size' in self.pooling_params.keys() else 1
            pool_stride = self.pooling_params['stride'] if 'stride' in self.pooling_params.keys() else pool_kernel
            pool_padding = self.pooling_params['padding'] if 'padding' in self.pooling_params.keys() else 0
            pool_dilation = self.pooling_params['dilation'] if 'dilation' in self.pooling_params.keys() else 1


            # the output features of pooling and convolution are calculated with same formula
            # activation does not modify the output size
            def layer_out_size(in_h, in_w, kernel, stride, padding, dilation):

                out_h = ( (in_h + 2*padding - dilation * (kernel -1) -1) / stride ) +1
                out_w = ((in_w + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1
                return floor(out_h), floor(out_w)

            in_channels, in_h, in_w, = tuple(self.in_size)
            N = len(self.channels)
            P = self.pool_every

            conv_block = 0  # conv+activation layer block
            pooling_layers = 0
            for i in range(N):
                in_h, in_w = layer_out_size(in_h, in_w, conv_kernel, conv_stride, conv_padding, conv_dilation)
                conv_block += 1
                if conv_block % P == 0 & pooling_layers < floor(N / P):
                    in_h, in_w = layer_out_size(in_h, in_w, pool_kernel, pool_stride, pool_padding, pool_dilation)
                    pooling_layers += 1

            in_classifier_features = int(in_h) * int(in_w) * self.channels[-1]
            return in_classifier_features
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_classifier(self):
        layers = []

        # Discover the number of features after the CNN part.
        n_features = self._n_features()

        # TODO: Create the classifier part of the model:
        #  (FC -> ACT)*M -> Linear
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======

        #first layer
        layers.append(nn.Linear(n_features, self.hidden_dims[0]))
        layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))

        for i in range(len(self.hidden_dims) -1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))

        #add final Linear layer
        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))

        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        out = 0

        out_ext_in_class = self.feature_extractor(x)
        out_ext_in_class = out_ext_in_class.view(out_ext_in_class.shape[0], -1)
        out = self.classifier(out_ext_in_class)

        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======

        #Main Path
        main_layers = []
        N = len(channels)

        for i in range(N -1):
            if i == 0: # first input layer
                # calculate padding assuming odd kernel size
                main_layers.append(nn.Conv2d(in_channels, channels[i], kernel_size=kernel_sizes[0], padding=( int((kernel_sizes[0]-1)/2) ) ))
                if dropout != 0.0: main_layers.append( nn.Dropout2d(dropout) )
                if batchnorm == True: main_layers.append(nn.BatchNorm2d(channels[i]))
            else: #middle layers
                padding = (kernel_sizes[i]-1) / 2
                main_layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=kernel_sizes[i], padding=( int(padding) ) ))
                if dropout != 0.0: main_layers.append(nn.Dropout2d(dropout))
                if batchnorm == True: main_layers.append(nn.BatchNorm2d(channels[i]))
            main_layers.append(ACTIVATIONS[activation_type]( **activation_params ))  # activation layer

        #last layer
        padding = (kernel_sizes[-1] - 1) / 2
        main_layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_size=kernel_sizes[-1], padding=( int(padding) ) ))
        self.main_path = nn.Sequential(*main_layers)


        shortcut_layers = []
        if in_channels == channels[-1]:
            shortcut_layers.append( nn.Identity() )
        else: #different input and output, add a convolution to project.
            shortcut_layers.append( nn.Conv2d(in_channels, channels[-1], kernel_size = 1, bias = False ) )
        self.shortcut_path = nn.Sequential(*shortcut_layers)
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        # ====== YOUR CODE: ======

        middle_channels = [ inner_channels[0] ] # first element of channels determines the initial projection
        kernels = [1] #kernel for initial projection

        for i in range( len(inner_channels) ):
            middle_channels.append( inner_channels[i] )
            kernels.append( inner_kernel_sizes[i] )

        middle_channels.append( in_out_channels )  # to project back to initial channel amount
        kernels.append(1) #kernel for last projection

        super().__init__(in_channels=in_out_channels, channels=middle_channels, kernel_sizes=kernels, **kwargs)

        # ========================


class ResNetClassifier(ConvClassifier):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        # ====== YOUR CODE: ======
        # Loop over groups of P output channels and create a block from them.
        kernel = 3
        N = len(self.channels)
        P = self.pool_every
        from math import floor

        for i in range(floor(N/P)):
            if i == 0:
                layers.append( ResidualBlock(in_channels, self.channels[0:P], [kernel] * P, batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type, activation_params=self.activation_params) ) # first input layer
            else:
                kernel_sizes = [kernel] * P
                channels = self.channels[i*P : (i+1)*P]
                layers.append(ResidualBlock(self.channels[(i - 1)*P], channels, kernel_sizes,
                                            batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type, activation_params=self.activation_params))
            layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))  # activation layer
            # Add pooling layer every P convolutional layers and up to floor(N/P)
            layers.append(POOLINGS[self.pooling_type](**self.pooling_params))

        #Final block with no Pooling
        additional = N%P
        if additional != 0:
            kernel_sizes = [kernel] * additional
            channels = self.channels[(floor(N/P))*P:]
            layers.append( ResidualBlock(self.channels[(floor(N/P)*P)-1], channels, kernel_sizes,
                                         batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type, activation_params=self.activation_params) )
            layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))  # activation layer

        # ========================
        seq = nn.Sequential(*layers)
        return seq

'''
class YourCodeNet(ConvClassifier):
    def __init__(self, *args, **kwargs):
        """
        See ConvClassifier.__init__
        """
        super().__init__(*args, **kwargs)

        # TODO: Add any additional initialization as needed.
        # ====== YOUR CODE: ======
        pass
        # ========================

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======

    # ========================
'''