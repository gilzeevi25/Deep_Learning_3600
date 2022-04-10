r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""



# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size']=53
    hypers['h_dim']=128
    hypers['z_dim']=32
    hypers['x_sigma2']=0.002
    hypers['learn_rate']=0.0001
    hypers['betas']=(0.9,0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**<br><br>
A parametric likelihood distribution was denoted as, $p _{\bb{\beta}}(\bb{X} | \bb{Z}=\bb{z}) = \mathcal{N}( \Psi _{\bb{\beta}}(\bb{z}) , \sigma^2 \bb{I} )$, where $\sigma^2$ represents the variance of the normal distribution.<br>
In general, Increasing $\sigma^2$ will yield a wider normal distribution with a cruve broader and shorter. if the variance is small (where<br> most values occur very close to the mean), the curve will be narrow and tall in the middle.
so from that statistical reasoning we conclude that increasing $\sigma^2$ will get us more 'variablitiy' in sampling - samples will probably
look different from the original dataset. on the other hand, a low value of $\sigma^2$ will yieald samples similar to the ones in the dataset.<br>
Furthermore, increasing $\sigma^2$ will result in less significat data-reconstruction loss $\rightarrow$ data-reconstruction loss decreases and the variance of the generated data will be large. on the contrary,  decreasing $\sigma^2$, the generated data becomes more 'biased' towards the training data.
"""

part2_q2 = r"""
**Your answer:**<br><br>
**1)**<br>
**<u>reconstruction loss</u>** - constructs the data to be as close as possible to the original input data using Mean Squared Error.<br>
minimizing it means we minimize the difference between the real observation $x$ and the VAE output (encoded and decoded) 
$\Psi _{\bb{\beta}}\left(  \bb{\mu} _{\bb{\alpha}}(\bb{x})  +
\bb{\Sigma}^{\frac{1}{2}} _{\bb{\alpha}}(\bb{x}) \bb{u}   \right)$<br><br>
**<u>KL divergence loss</u>** - approximates the latent space distribution to be as close to some known informative distribution like the gaussian/normal distribution

**2)**<br>
VAE tries tp estimate the evidence distribution $p(X)$, but this is a hard task so we maximize a lower bound on $log(p(X))$.<br>
The lower bound denoted by  $ \log p(\bb{X}) \ge \mathbb{E} _{\bb{z} \sim q _{\bb{\alpha}} }\left[ \log  p _{\bb{\beta}}(\bb{X} | \bb{z}) \right]
-  \mathcal{D} _{\mathrm{KL}}\left(q _{\bb{\alpha}}(\bb{Z} | \bb{X})\,\left\|\, p(\bb{Z} )\right.\right)
$
where $
\mathcal{D} _{\mathrm{KL}}(q\left\|\right.p) =
\mathbb{E}_{\bb{z}\sim q}\left[ \log \frac{q(\bb{Z})}{p(\bb{Z})} \right]
$
is the Kullback-Liebler divergence.<br>
BY minimizing KL-loss, we seek a gaussian distribution $q(Z)$ that is as close as possible to $p(Z)$ in order to tighten up the lower bound.<br>
**3)**<br>
The benefit is the possibility of controllong the variability of the latent space distribution. it is then can be witnessed as a tradeoff <br>
between reconstruction loss and KL-loss, and presented in part **1)** of this question.
"""

part2_q3 = r"""
**Your answer:**<br><br>
we start by maximizing the evidence  distribution, $p(\bb{X})$, because we dont have that evidence distribution.
we estimate it in order to generate new data and by doing so, we are tightening up the lower bound $ \log p(\bb{X}) \ge \mathbb{E} _{\bb{z} \sim q _{\bb{\alpha}} }\left[ \log  p _{\bb{\beta}}(\bb{X} | \bb{z}) \right]
-  \mathcal{D} _{\mathrm{KL}}\left(q _{\bb{\alpha}}(\bb{Z} | \bb{X})\,\left\|\, p(\bb{Z} )\right.\right)
$ hence getting
the estimator of distribution, to be as close as we can to $p(\bb{X})$.

"""

part2_q4 = r"""
**Your answer:**<br><br>
taking the log not only simplifies the subsequent mathematical analysis by reducing multipication and division to addition and subtraction, but it also helps numerically because the product of a large number of small probabilities can easily underflow the numerical precision of the computer, and this is resolved by computing instead the sum of the log probabilities.
In addition, the log function is differentiable in all the range it's defined upon, which allows using the derivative easily .


"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=32,
        z_dim=15,
        data_label=1,
        label_noise=0.2,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            betas=(0.5, 0.999)
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            betas=(0.5, 0.999)
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
#     hypers['batch_size']=53
    hypers['batch_size']=32
    hypers['z_dim']=8
    hypers['data_label']=1
    hypers['label_noise']=0.1
    hypers['discriminator_optimizer']['type']='Adam'
#     hypers['discriminator_optimizer']['weight_decay']=0.001
    hypers['discriminator_optimizer']['betas']=(0.5, 0.999)
    hypers['discriminator_optimizer']['lr']=0.0002
    hypers['generator_optimizer']['type']='Adam'
#     hypers['generator_optimizer']['weight_decay']=0.001
    hypers['generator_optimizer']['betas']=(0.3, 0.999)
    hypers['generator_optimizer']['lr']=0.0002
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**<br><br>
The generator training process relies on sampling/generating data and then presenting it to the discriminator and then calculating the loss.<br>
In this process we do want the generator weights to be updated hence we do save gradients in order to use back-propagation.<br>
on the other hand, the discriminator simply behaves as a classifier,<br>
and during that process we keep the generator constant, not saving the generator gradients,<br> thus enabling the discriminator to learn the generator as is and its "behaviour".


"""

part3_q2 = r"""
**Your answer:**<br><br>
**1)**<br>
No, we shouldn't stop training solely based on low Generator loss!<br>
The reason lies in the fact that we're trying to reach an equilibrium between the generator loss and the discriminator loss.<br>
If the discriminator isn't accurate enough and the generator loss is very low, it simply means that the generator is performing well in fooling the discriminator.<br>
Hence, if the Generator loss is low, we cannot conclude on the performance of the entire model due to dependency between the discriminator and generator.

**2)**<br>
we have two possible interpretationts:<br><br>
**a.** If discriminator loss is temporal stuck it might mean the discriminator is ahead of the Generator in the learning process.<br>
 the descriminator tells the difference between real and fake images thus forcing the generator to keep learning the discriminator behaviour in order to "catch up".<br>

**b.** If discriminator loss is stuck permanently in its learning process and will not improve:<br> 
The generator is improving. It is getting better on generating fake images that the discriminator is not being able to discriminate.<br>
The discriminator loss stuck at constant means it is not improving, so its accuracy for discriminating fake images from real ones is still the same,<br>
it might be stuck in a local minimum.

"""

part3_q3 = r"""
**Your answer:**<br><br>
The VAE model yielded more blurry with less sharp edges which focuses on the foreground of the images, while GAN model yielded sharper images with more features and colors<br>
hence focusing also on the background.<br>
VAE has a term of reconstruction loss in the general loss function, which forcing the output to be similar to the input with applying MSE loss.<br>
Moreover, The process of "generation" in VAE is done by an encoder and decoder, which makes it lose quality and not get that better results we see in GAN.<br>
it results in smooth, blurry images, and the generated images looks more similiar to each other.<br>
When engaging the GAN model on the other hand, the generator does not have 'direct access' to real images,<br>
but learns how those should look through the decisions of the discriminator,forcing it's predictions to be more realistic. <br>
the generator at the beginning can easily spot whats fake and whats real, and by the end of the learning process, **hopefully**,<br>
the discriminator tend to random choice, hence its hard to tell for the discriminator whether its a fake or true image and thus makes the output more realistic.

"""

# ==============
