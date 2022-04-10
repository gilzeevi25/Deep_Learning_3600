
PART4_CUSTOM_DATA_URL = None

def vanilla_hyperparams():
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
    hypers['batch_size']=32
    hypers['z_dim']=8
    hypers['data_label']=1
    hypers['label_noise']=0.1
    hypers['discriminator_optimizer']['type']='Adam'
    hypers['discriminator_optimizer']['betas']=(0.5, 0.999)
    hypers['discriminator_optimizer']['lr']=0.0002
    hypers['generator_optimizer']['type']='Adam'
    hypers['generator_optimizer']['betas']=(0.3, 0.999)
    hypers['generator_optimizer']['lr']=0.0002
    # ========================
    return hypers

  
def sn_gan_hyperparams():
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
    hypers['batch_size']=32
    hypers['z_dim']=8
    hypers['data_label']=1
    hypers['label_noise']=0.1
    hypers['discriminator_optimizer']['type']='Adam'
    hypers['discriminator_optimizer']['betas']=(0.5, 0.999)
    hypers['discriminator_optimizer']['lr']=0.0002
    hypers['generator_optimizer']['type']='Adam'
    hypers['generator_optimizer']['betas']=(0.3, 0.999)
    hypers['generator_optimizer']['lr']=0.0002
    # ========================
    return hypers

def wgan_hyperparams():
    hypers = dict(
        batch_size=32,
        z_dim=8,
        discriminator_optimizer=dict(
            type="RMSprop",  # Any name in nn.optim like SGD, Adam
            lr=0.0005
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="RMSprop",  # Any name in nn.optim like SGD, Adam
            lr=0.0005
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size']=32
    hypers['z_dim']=8
    hypers['discriminator_optimizer']['type']='RMSprop'
    hypers['discriminator_optimizer']['lr']=0.0005



    hypers['generator_optimizer']['type']='RMSprop'
    hypers['generator_optimizer']['lr']=0.0005
    hypers['n_critic'] = 5 #number of updates on discriminator per update of generator
    hypers['c'] = 0.01 #for clipping
    # ========================
    return hypers