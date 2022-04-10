import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn.utils.spectral_norm as SN
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        modules = []
        # Based on architecture guidelines of https://arxiv.org/pdf/1511.06434.pdf page 3
        # 3 inner layers of: Conv (5x5 kernel, , stride=2, padding=2), Batchnorm, LeakyRelu, channels 64, 128, 256 respectively, no hidden FC layers
        # Add linear layer for output
        in_channels = in_size[0]
        channels = [64, 128, 256]

        
        modules.append(SN(nn.Conv2d(in_channels, channels[0],kernel_size=5, stride=2, padding=2)))
        modules.append(nn.BatchNorm2d(channels[0]))
        modules.append(nn.ReLU())

        modules.append(SN(nn.Conv2d(channels[0], channels[1],kernel_size=5, stride=2, padding=2)))
        modules.append(nn.BatchNorm2d(channels[1]))
        modules.append(nn.ReLU())

        modules.append(SN(nn.Conv2d(channels[1], channels[2], kernel_size=5, stride=2, padding=2)))
        modules.append(nn.BatchNorm2d(channels[2]))
        modules.append(nn.ReLU())
        self.disc_cnn = nn.Sequential(*modules)

        # output layer - flatten features
        num_features = torch.zeros(in_size).unsqueeze(0)
        num_features = self.disc_cnn(num_features).view(1, -1)
        self.disc_fc = nn.Linear(num_features.shape[1], 1)
        

        # ========================


    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        N = x.shape[0]
        z = self.disc_cnn(x)
        z = z.view(N, -1)
        y = self.disc_fc(z)
        
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        #hint (you dont have to use....)
        # input to Generator must be a sample u from latent space
        # output is x' with same shape as x, to be input to Discriminator
        modules = []
        self.featuremap_size = featuremap_size
        self.out_channels = out_channels

        # Based on architecture guidelines of https://arxiv.org/pdf/1511.06434.pdf page 3 and 4
        # 4 layers of: Reverse Conv (5x5 kernel, , stride=2, padding=2), Batchnorm, LeakyReLU (tanh for output layer)
        # inner channels 512, 256, 128 respectively, no hidden FC layers
        # assume images of fixed size 64x64.
        leaky_slope = 0.2
        inner_conv_channels = [512, 256, 128]
        self.in_channels = 1024

        projection_dimension = self.in_channels * self.featuremap_size * self.featuremap_size
        self.gen_fc = nn.Linear(z_dim, projection_dimension, bias=False)

        modules.append(nn.ConvTranspose2d(self.in_channels, inner_conv_channels[0], kernel_size=5, stride=2, padding=2, output_padding=1))
        modules.append(nn.BatchNorm2d(inner_conv_channels[0]))
        modules.append(nn.LeakyReLU(negative_slope=leaky_slope))

        modules.append(nn.ConvTranspose2d(inner_conv_channels[0], inner_conv_channels[1], kernel_size=5, stride=2, padding=2, output_padding=1))
        modules.append(nn.BatchNorm2d(inner_conv_channels[1]))
        modules.append(nn.LeakyReLU(negative_slope=leaky_slope))

        modules.append(nn.ConvTranspose2d(inner_conv_channels[1], inner_conv_channels[2], kernel_size=5, stride=2, padding=2, output_padding=1))
        modules.append(nn.BatchNorm2d(inner_conv_channels[2]))
        modules.append(nn.LeakyReLU(negative_slope=leaky_slope))

        # output layer
        modules.append(nn.ConvTranspose2d(inner_conv_channels[2], self.out_channels, kernel_size=5, stride=2, padding=2, output_padding=1))
        modules.append(nn.Tanh())


        # ========================
        self.gen_cnn = nn.Sequential(*modules)

        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        z_samples = torch.randn( (n, self.z_dim), device=device)
        if with_grad == True:
            samples = self.forward(z_samples)
        else:
            samples = self.forward(z_samples).detach()

        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        N = z.shape[0]
        flat_z = self.gen_fc(z)
        image_size = self.in_channels, self.featuremap_size, self.featuremap_size #C, H, W
        batch_images = tuple([N]) + image_size
        projected_reshaped_z_samples = torch.reshape(flat_z, batch_images)
        x = self.gen_cnn(projected_reshaped_z_samples)
        
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    eps_noise = label_noise / 2

    real_label_noise = torch.FloatTensor(y_data.shape).uniform_(-eps_noise, eps_noise) #CHANGED y_data.shape[0] TO y_data.shape
    real_labels = data_label + real_label_noise

    generated_label_noise = torch.FloatTensor(y_generated.shape).uniform_(-eps_noise, eps_noise) #CHANGED y_generated.shape[0] TO y_generated.shape
    generated_labels = (1 - data_label) + generated_label_noise

    criterion = torch.nn.BCEWithLogitsLoss() #cross entropy
    loss_data = criterion(y_data, real_labels.to(device=y_data.device))
    loss_generated = criterion(y_generated, generated_labels.to(device=y_generated.device))
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    #All labels are the same value.
    generated_labels = torch.ones(y_generated.shape, dtype=torch.float) * data_label #CHANGED y_generated.shape[0] TO y_generated.shape
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(y_generated, generated_labels.to(device=y_generated.device))
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    samples = gen_model.sample(x_data.shape[0], with_grad=False)
    gen_pred = dsc_model(samples)
    y_pred = dsc_model(x_data)

    dsc_optimizer.zero_grad()
    dsc_loss = dsc_loss_fn(y_pred, gen_pred)
    dsc_loss.backward()
    dsc_optimizer.step()

    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    samples = gen_model.sample(x_data.shape[0], with_grad=True)
    gen_data = dsc_model(samples)
    gen_optimizer.zero_grad()
    gen_loss = gen_loss_fn(gen_data)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    if not(len(gen_losses) % 10) or not(len(gen_losses) % 99):
        saved = True
        if saved and checkpoint_file is not None:
            torch.save(gen_model, checkpoint_file)
    # ========================

    return saved
