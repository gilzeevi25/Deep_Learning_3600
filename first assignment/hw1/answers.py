r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1) **FALSE** - The test set allows us estimate our model performance on an unseen data hence to test how 'successful' he is on "real" data.
 on the other hand the validation set helps us to estimate the our in-sample error.
2) **FALSE** - not any split will do, because if there exists a split that splits the data such that a one (or more) specific label
 can reside only in test data and not in the train also, our model wont be able to learn that labels thus
  failing to this instances which that unique to test only
3) **TRUE** - the test should not be exposed while performing training task.
    the test set will only evaluate the performance of the model on an unseen data,
     thus it should remain unbiased and not to be exposed in the learning process.
4) **FALSE** - we use the validation-set performance of each fold to determine which of the training hyperparameters performes the best, hence tuning the 
    model to perform as best as it can be and reduce the 'in sample error', not the generalization error. 
    
    
 
"""

part1_q2 = r"""
**Your answer:**
 **The main approach is justified but the delivery is wrong** - the part which isnt justified is the part where the friend tuned a hyperparameter of the model 
 using the test set instead of using the validation set.
 this will lead to  overfittting to a specific test set,
 and will deteriorate the generalization of the model.

                     
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
In general, when $K_{nn} = 1$ , the model's prediction is based on a single sample - one closest neighbor. this heuristic is not that robust due to the fact that it is very sensitive to noise, outliers, mislabeling of data. by increasing $K$, the robustness of the model is increased, but only to a certain point which from then on, the model generalization will deteriorate. for the opposite case, when $K$ is a big number, or even  $K = n_{samples}$ , then we might receive an overlapping between classes or just a pretty bad generalized model which predict based on the majority of labels.
So, to sum thing up, we want to increase k, but not that much. how much? thats why why want to use k-fold CV in order to understand how much to increase, its never an easy automatic answer unfortunately
"""

part2_q2 = r"""
**Your answer:**
In general, K-fold CV is a common way to tune the hyperparameters of the model such that it most likely will improve its generalization. by applying CV to different 'folds' of the data we can avoid overfitting to a specific trend in the data. on the other hand, the following cases cause overfitting and loss of generalization in these kind of ways:<br><br>
    **1)** using different models on the same training set, just in order to maximize accuracy will just cause the model to overfit to the training data and to badly affect the generalization of the model. instead of trying to find the model that generalizes the best, this method is simply trying to fit as good as it gets to the training data<br><br>
    **2)** using different models on the same data, and choosing the best in respect to high test prediction accuracy will cause the model to overfit to a specific test data.
    the generality of the model will be damaged and the model will poorly perform on other real data, due to the fact that we maximized the model to bestly fit to a certain data.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
The selection of parameter delta is arbitrary as it is not a hyperparameter for tuning the model, optimizing it and its performance. We require it to be positive as it is our guarantee that we label correctly the sample: we will assign a higher score to the correct class by at least $\delta$ distance with respect to the other classes.<br><br>
With this understanding, we can state that delta is also the margin and hence is directly related $\lambda$. By fixing arbitrarily $\delta$, we can use $\lambda$ as hyperparameter to adjust to the loss.
"""

part3_q2 = r"""
**Your answer:**
1. If we observe the image representation of the weight matrices, we can observe they are very close and similar to each of the digits (first element is 0, second representation of weight matrix is 1, and so on).
As per the weight matrixes represented as images, we observe that they are quite homogeneous but present differences in colours where the lines of the digit should be. When predicting, we multiply the new sample with weight matrix and obtain, and evaluate which weight matrix is closer to the input - it will get the highest score.
What the classifier is doing, is placing the sample, weight it by the weight matrix and evaluate on which side of the hyperplane it is (to which digit is closer to).
This explains why some samples are misclassified. Lets take for example, the first digit '5' to appear. Its trace is quite similar to digit '6' and not a traditional drawing of '5', which will make it closer to the side of the hyperplane of the domain of class of digit '6'.<br><br>

2. The interpretation is different from KNN mainly because for linear regression, the model learns a weight matrix and defines a hyperplane, SVM tries to generalize and learn the particularities and representation of each class. It will use it to place the sample in the correct side and therefore class. 
In KNN, we will check and compare the new sample with each and all training data, and evaluate the majority of the closest ones. 
If we consider previous analysis about weight matrix as image representation, KNN might me interpreted or considered similar in the sense that we analyze how similar is the new sample to the previous images (not the weight matrix but each of the images per class).

"""

part3_q3 = r"""
**Your answer:**
1. The chosen rate is between good and high. After 2 to 5 epochs it already gets close to the convergence zone and after 10 epochs it almost converges to the final loss achieved. 
A too small learning rate will output a curve that does not decrease sufficiently and will not converge in the loss, defining a model with not as good accuracy, at least with the same amount of epochs. 
A too high learning rate would take big steps and might not get close to a minimum during optimization, and not reducing the loss within the iteration of epochs <br><br>

2. Based out of the accuracy plot, we consider the model to be slightly overfitted to the training set. This is because we are achieving a very high accuracy on training but a slight less accuracy on training.
It would be a better balance if we could lower the training accuracy and increase the validation accuracy, meaning we overfit less to training data and generalize the model better.
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
The desired pattern we want to achieve is when  $y^{(i)} - \hat{y}^{(i)}=0$ or as close to that as we can get.
In that case, we will witness how the predictions reside on ( or at least very near)  an horizontal line at $y_{axis} = 0$ and that is our goal.
We can see how in the final plot after CV, the residuals less scatter and more uniformly reside near the $y_{axis} = 0$ which means our model better generalized 
than the top-5 features plot, where we witness a wider scatter and many outliers outside the margins of the $y_{axis} = 0$ horizontal line.
Another notable thing that the outlier residuals in the final plot after CV still lay around the margins of the horizontal line hence
our generalized model was able to describe the data alot better. 
"""

part4_q2 = r"""
**Your answer:**<br>
**1)**  We must firstly recall that our general task is to solve the linear equation $\hat{y} = \vectr{w}\vec{x} + b$.
        Even though we transform the features to a higher dimension,
         our main purpose is still adressing the linear operation of matrix multipication between the weights and the features
         thus its still a linear regression model, only now the features are more 'informative' and can describe the data better.<br><br>
**2)** We can try and fit any non-linear transformation to raise dimensionality as much we like, by theory. But in practice, the more we increase the
       dimensionality, we risk in getting less efficient in terms of runtime and sparsity<br><br>
**3)** By 'Adding' non-linear features we actually increase the dimension, and then in this higher dimension $N$ we seperate the data with
       a hyperplane which is an $N-1$ dimensional object (or rather, 'manifold') and can be mathematically described by a vector orthogonal to the hyperplane.
       So to sum up, be increasing dimensions of feature, we respectfully will have to increase the hyperplane dimensions to $N-1$ in order to apply the decision boundry.
         
            
"""

part4_q3 = r"""
**Your answer:**<br>
**1)** By using logspace we actually get to inspect equally spaced $\lambda$ on a logscale.
       we suspect that different $\lambda$ by magnitude can affect the learning process in all sorts of way (overshooting for example).
       So, the logspace step is better just than the linear equaly space in a matter
       that we get to inspect different orders of magnitude of $\lambda$ in less iterations, thus to perform faster on our CV.<br><br>
**2)** We notice that degree range is 3, $\lambda$ range is 20, and we are using 3 - fold CV so in total 120 times
        $$ 3_{degree}X3_{kfold}X20_{\lambda} = 120_{times} $$
        
"""

# ==============
