import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
import tqdm
import sys
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
'''
This file's purpose is to compute Inception Score (IS) of generated images by GAN's generator.

*** NOTE:
It Has been inspired from Shane Barratt's git https://github.com/sbarratt/inception-score-pytorch
and has been devised for our project's purposes 
***
'''
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """
    Computes the inception score of the generated images imgs
    imgs -- Gan's generated imgs - we use 1000 images
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """

    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear',align_corners=True).type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def plot_inception(models,names,epochs = 100,nums = {0:1,1:2,2:5,3:10,4:20},nums_sn = {0:1,1:2,2:5,3:10,4:20},k=3):
    '''
    plotting the inception scores with additional trend curve plot using polynomial splines
    '''
#     models = load_inception_results()
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2 , figsize=(18, 7))
    for name in names:
        if name == 'gan' or name == 'gan_sn':
            y = models[name][0]
            x = np.arange(1,len(y)+1,1)
            xs, ys = interpolate_graph(x,y,k,len(y))
            ax1.plot(x,y,label = f'{name} model')
            ax1.set_title('Inception Score',size=18)
            ax1.set_xlabel('Epoch',fontsize=14)
            ax1.set_ylabel('Inception',fontsize=14)
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09),
          fancybox=True, shadow=True, ncol=5)
            ax2.plot(xs,ys,label = f'{name} model')
            ax2.set_title('Inception Trend',size=18)
            ax2.set_xlabel('Epoch',fontsize=14)
            ax2.set_ylabel('Inception',fontsize=14)
            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09),
          fancybox=True, shadow=True, ncol=5)
        if name == 'wgan' or name == 'wgan_sn':
            for i in range(len(models[name])):
                y = models[name][i]
                x = np.arange(1,len(y)+1,1)
                xs, ys = interpolate_graph(x,y,k,len(y))
                ax1.plot(x,y,label = f'{name}: n_critic of {nums[i]}') if name == 'wgan' else ax1.plot(x,y,label = f'{name}: n_critic of {nums_sn[i]}')
                ax1.set_title('Inception Score',size=18)
                ax1.set_xlabel('Epoch',fontsize=14)
                ax1.set_ylabel('Inception',fontsize=14)
                ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09),
              fancybox=True, shadow=True, ncol=2)
                ax2.plot(xs,ys,label = f'{name}: n_critic of {nums[i]}') if name == 'wgan' else ax2.plot(xs,ys,label = f'{name}: n_critic of {nums_sn[i]}')
                ax2.set_title('Inception Trend',size=18)
                ax2.set_xlabel('Epoch',fontsize=14)
                ax2.set_ylabel('Inception',fontsize=14)
                ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09),
              fancybox=True, shadow=True, ncol=2)

    plt.show()

   
    
def interpolate_graph(x,y,k=3,epochs = 100):
    xs = np.linspace(0, x[-1], epochs)
    s = UnivariateSpline(x, y,k=k)
    ys = s(xs)
    return xs, ys
    
    