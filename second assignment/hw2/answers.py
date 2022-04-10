r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**<br>
**1.** as noted in the notebook that
the term $\pderiv{\vec{z}}{\vec{w}}$ is a 4D Jacobian if both $\vec{z}$ and $\vec{w}$
are 2D matrices so for the case mention above , producing the Jacobian from an output of $\mat{Z}_{[128,2048]}$ 
into an input of $\mat{X}_{[128,1024]}$ will require a $\mat{Jacobian}_{\large [\bf 128,2048,128,1024]}$
<br><br>
**2.** We know that $32_{bits} = 4_{bytes} = 2^2 {bytes}$. 
        Each cell of the Jacobian is represented by $2^2 {bytes}$ hence we denote the number of Gigabyte RAM required to store
        the Jacobian by the following formula:
        $$ \large [\bf 128,2048,128,1024] X 2^2_{bytes} = 2^2\cdot 2^7\cdot 2^{11}\cdot 2^7\cdot 2^{10} = 2^7\cdot 2^{30} bytes =
        \\ =\Large 128 GB$$

"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1,0.1,0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.1, 0.05, 0.007, 0.0002, 0.001
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.00053448133892335753 # acquired by cross validation, using np.logspace -> this is why the wierd number with many floating points
    # ========================
    return dict(wstd=wstd, lr=lr)

# Explain the graphs of no-dropout vs dropout. Do they match what you expected to see?
#
# If yes, explain why and provide examples based on the graphs.
# If no, explain what you think the problem is and what should be modified to fix it.
part2_q1 = r"""
**Your answer:**<br>
**1.** In general, dropout is a regularization technique for reducing overfitting  by preventing complex co-adaptations on training data.
So, where there is no dropout, we see how our model greatly perform on the train set but then perform not that great on the test set.
this is exactly what that dropout technique tries to handle - the overfitting of the learning process.
NEVERTHELESS! We expected to see better performance on the test for a dropout rate of 0.4, because this value sounds as an logical value to pick,
in order top drop some of the layers, but not that much and thus perform better in the learning process.
Furthermore, surprisingly enough, the high rate dropout of 0.8 performed really well on the train set, and performed on the test as good as the zero rate dropout,
and even slightly better. this caught us by surprise because we couldnt expect that loosing so many neurons on the learning process,
could lead to better performance in generalization than the "optimal" drop rate which we thought to be 0.4.
<br>
in our opinion there might be several reasons for the unexpected results:
 - We might screwed the dropout implementating and were mistaking in scaling the train time activation, 
 or used the bernoulli distribution not the way it should've been.
 - Maybe this perticular data, should be considered to train with much less neurons, to achieve better results.
 - We noticed that the Momentum optimizer is used for the dropout. maybe the momentum hyperparameter should be tweaked simultaneously with
   the parameters we tweaked in part2_dropout_hp the make the perforamnce of the learning better, and to get more reliable result, 
   with much more stable loss learning curve.<br>
<br>
**2)** Again, as mentioned in section 1 above, we expected to see better performance for a reasonable droprate, such as 0.4,
and worse performance for such a high droprate as 0.8.<br>
instead, we so just the opposite around, which took us by surprise.
furthermore, we see a more stable loss learning process for lower droprate 0.4 , then the 0.8 rate which the loss function is decreasing in a very noisy unstable way.
If we had more time, we would really love to keep tweaking with the model to find out if its possible to make 0.4 dropout rate, to out perform
non drop rate and 0.8 drop rate.
"""

part2_q2 = r"""
**Your answer:**<br>
**Yes!,strangely enough it is possible.**<br>
Binary cross entropy compares each of the predicted probabilities to actual class output. 
It then calculates the score that penalizes the probabilities based on the distance from the expected value.
the accuracy on the other hand, only "counting" the correct predictions,
Hence, the loss is more influenced by wrong predictions i.e. outliers rather than correct predictions.
Due to that, it is possible that for some class we're having bad predictions thus the loss increased.
Nevertheless it definitely can occur in parallel that in general the overall prediction of the other classes,
is high thus the accuracy increases also.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**<br>
The regular left block ( with a channel of 256 instead of 64) has 2 conv layers, applying a $[3X3]_{kernel}$ each, hence: 
$$ parameters_{regularBLOCK} => 2 \cdot [(3^{2}\cdot 256 +1)\cdot 256]  = 1,180,160_{parameters}$$
while on the otherhand the bottleneck block has several steps to maintain:
 1) $1X1X256 => 64$ - additional 1x1 convolution to project the channel dimension.
 2) $3X3X64 => 64$ 
 3) $1X1X64 => 256$ - another projection to get back to the original channel
hence the bottleneck consist the following number of parameters:
$$ parameters_{botllnckBLOCK} => (1^{2}\cdot 256+1)\cdot 64+(3^{2}\cdot 64+1)\cdot 64+(1^{2}\cdot 64+1)\cdot 256  = 70,016_{parameters}$$<br>
**due to the calculations above we witness the really significant parameters reduction**
<br>
In terms of qualitative assessment,we'll assume input size  of $(256, H, W)$,and the relu operations as $ 256\cdot H\cdot W$<br>
So,the number of floating point operations will , roughly , be: <br><br>
$\bullet regular\,block = (1,180,160 + 256)\cdot H\cdot W = 1,180,416\cdot H\cdot W_{operations}$<br>
$\bullet ottleneck\,block = (70,016 + 256)\cdot H\cdot W = 70,272\cdot H\cdot W_{operations}$<br>
<br>
while the regular block affects the input spatially by maintaining the number of feature maps,
 the bottleneck block affect both spatial and across feature map of the input by using a projection into smaller feature map and then another projection into
 original channel and thus reducing parameters significantly.<br>
 
"""

part3_q2 = r"""
**Your answer:**<br>
We can conclude the following:
* The best results are with $L = 2$ and $L = 4$.
* with $L = 8$ we witness a deterioration in terms of performance<br>
where the accuracy decreases and the loss in general is minimized worse than $L = 2$ and $L = 4$.
* the interesting part in our opinion is that we see that for $L = 16$ the model "stop learning", and the network becomes untrainable.
* we can generally conclude from experiment 1.1 that the "deeper" we go, we risk in "diverging" in training terms and failing to build a generalized model.
* the reason for this can be vanishing gradient problem and it affects the training in deeper network.
* The solution for this is of course the ResNet we implemented. <br>The skip connections in ResNet allow gradient information to pass through the layers, by creating "highways" of information,<br>
where the output of a previous layer/activation is added to the output of a deeper layer
,hence dealing with vanishing gradient.
"""

part3_q3 = r"""
**Your answer:**<br>
From this experiment we can reinforce our conclusions from the previous experiment and witness how
lower depth networks train better and achieve higher accuracy. in addition we see that for $L = 4$ <br>
some combinations almost achieved 90% accuracy, with pretty much low batch size (30) and not that much epochs (25) which is really really
nice to see:)
<br>
furthermore we notice that, in general the performance improve as we increase the number of the filter,
and it makes sense because higher number of filters allows better features extractions from an image.
"""

part3_q4 = r"""
**Your answer:**<br>
We still witness the same phenomena - as the networks gets deeper the networks fails to train. it can be of course due to vanishing gradients<br>
but here there is another factor to take into considertaion, the kept growing Filters (Kernels).
it might be that the transitions from large descriptors to very small makes it harder to identify the meaningful descriptors for the network, thus making the network untrainable.<br>
But nevertheless, we achieved really high test accuracy with $L=1$ which is really nice and intresting to see, whereas the 
epochs and batch size are relatively small.

"""

part3_q5 = r"""
**Your answer:**<br>
Okay, so as expected, the ResNet architecture solves the probelm of the vanishing gradient thus enables us to train deeper netowrks.
although with the chosen epochs and batch size which were relatively small the models dont reach that high of accuracy,
as previous experiment but our opinion is that if we train the models of experiment 4 with more batch size and more epochs, 
we will even generalize better than the previous experiments, and maybe prevent the overfitting we might be spotting in the CNN architecture

"""

part3_q6 = r"""
**Your answer:**

"""
# ==============
