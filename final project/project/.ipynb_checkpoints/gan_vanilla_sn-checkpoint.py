import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SN
from torch import Tensor
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import pathlib
import unittest
import os
import sys
import pathlib
import urllib
import shutil
import re
import zipfile
from typing import Callable
import numpy as np
import cs3600.plot as plot
import cs3600.download
from hw4.answers import PART3_CUSTOM_DATA_URL as CUSTOM_DATA_URL
from hw4.answers import part3_gan_hyperparams
import matplotlib.pyplot as plt



def load_bush_dataset(im_size = 64,to_plot=True):
    '''
    A local function to load George W Bush dataset based on GAN notebook
    input: 
        fixed image size (default: 64)
        whether to plot a sample of the database or not
    return: 
        Bush dataset
    '''
    DATA_DIR = pathlib.Path.home().joinpath('.pytorch-datasets')
    if CUSTOM_DATA_URL is None:
        DATA_URL = 'http://vis-www.cs.umass.edu/lfw/lfw-bush.zip'
    else:
        DATA_URL = CUSTOM_DATA_URL
    _, dataset_dir = cs3600.download.download_data(out_path=DATA_DIR, url=DATA_URL, extract=True, force=False)
    im_size = 64
    tf = T.Compose([
        # Resize to constant spatial dimensions
        T.Resize((im_size, im_size)),
        # PIL.Image -> torch.Tensor
        T.ToTensor(),
        # Dynamic range [0,1] -> [-1, 1]
        T.Normalize(mean=(.5,.5,.5), std=(.5,.5,.5)),
    ])
    ds_gwb = ImageFolder(os.path.dirname(dataset_dir), tf)
    print(f'Found {len(ds_gwb)} images in dataset folder.')
    if to_plot:
        _ = plot.dataset_first_n(ds_gwb, 50, figsize=(15,10), nrows=5)
    return ds_gwb 



def load_vanilla_GAN(ds_gwb,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),z_dim = 128,n=15,SN_act=False):
    hp = part3_gan_hyperparams()

    checkpoint_file_final = 'checkpoints/gan_final.pt'
    
    dsc = Discriminator(in_size=ds_gwb[0][0].shape,SN_active=SN_act).to(device)
    print(f"*** Chosen Discriminator Architecture for Vanilla GAN: *** \n\n{dsc} \n\n")
    gener = Generator(z_dim, 4).to(device)
    print(f"*** Chosen Generator Architecture for Vanilla GAN: *** \n\n{gener}\n\n")
    print(f"*** Chosen HyperParameters for Vanilla GAN: *** \n\n{hp}\n\n")
    print('*** Sampled Images Generated from Vanilla GAN: ***')
    gen = torch.load(checkpoint_file_final, map_location=device)
    samples = gen.sample(n, with_grad=False).cpu()
    fig, _ = plot.tensors_as_images(samples, nrows=3, figsize=(6,6))
    return gen
    
def load_SN_GAN(ds_gwb,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),z_dim = 128,n=15,SN_act=True):
    hp = part3_gan_hyperparams()

    checkpoint_file_final = 'checkpoints/SNgan.pt'

    dsc = Discriminator(in_size=ds_gwb[0][0].shape,SN_active=SN_act).to(device)
    print(f"*** Chosen Discriminator Architecture for SNGAN: *** \n\n{dsc} \n\n")
    gener = Generator(z_dim, 4).to(device)
    print(f"*** Chosen Generator Architecture for SNGAN: *** \n\n{gener}\n\n")
    print(f"*** Chosen HyperParameters for SNGAN: *** \n\n{hp}\n\n")
    print('*** Sampled Images Generated from SNGAN: ***')
    gen = torch.load(checkpoint_file_final, map_location=device)
    samples = gen.sample(n, with_grad=False).cpu()
    fig, _ = plot.tensors_as_images(samples, nrows=3, figsize=(6,6))
    return gen

def compare_imgs(device):
    print('*** Sampled Images Generated from Vanilla GAN: ***')
    gen = torch.load('checkpoints/gan_final.pt', map_location=device)
    samples = gen.sample(15, with_grad=False).cpu()
    fig, _ = plot.tensors_as_images(samples, nrows=3, figsize=(6,6))
    plt.show()
    print('*** Sampled Images Generated from SNGAN: ***')
    gen = torch.load('checkpoints/SNgan.pt', map_location=device)
    samples = gen.sample(15, with_grad=False).cpu()
    fig, _ = plot.tensors_as_images(samples, nrows=3, figsize=(6,6))
    
    
class Discriminator(nn.Module):
    def __init__(self, in_size,SN_active=False):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        self.sn = SN_active
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

        modules.append(self.SN_activation(nn.Conv2d(in_channels, channels[0],kernel_size=5, stride=2, padding=2)))
        modules.append(nn.BatchNorm2d(channels[0]))
        modules.append(nn.ReLU())

        modules.append(self.SN_activation(nn.Conv2d(channels[0], channels[1],kernel_size=5, stride=2, padding=2)))
        modules.append(nn.BatchNorm2d(channels[1]))
        modules.append(nn.ReLU())

        modules.append(self.SN_activation(nn.Conv2d(channels[1], channels[2], kernel_size=5, stride=2, padding=2)))
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
    def SN_activation(self,module):
        '''
        If true applying SN on module
        :param module: Input of given module.
        :return: the Spectral Norm of a module or the module itself
        
        '''
        if self.sn:
            return SN(module)
        return module


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
