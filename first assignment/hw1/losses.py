import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: =====
        row_range = range(x_scores.size()[0])
        correct_class_score = x_scores[row_range,y].reshape(-1, 1)
        M = x_scores + self.delta - correct_class_score #marginal loss matrix - do not penalized correctly classified
        M[row_range,y] = 0 #remove delta on the correctly classified
        sample_loss = torch.max( torch.zeros(x_scores.shape), M) #choose max loss per sample

        loss = sample_loss.sum() / x.size()[0]
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx['row_range'] = row_range
        self.grad_ctx['M'] = M
        self.grad_ctx['x'] = x
        self.grad_ctx['y'] = y
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        G = torch.zeros(self.grad_ctx['M'].shape) # as per hint, x.T * G will give the gradients.
        # Need to prepare G for cases "correct class j == y_i" and "incorrect class j != y_i"

        G[ self.grad_ctx['M'] > 0 ] = 1 # these 1's are the cases of incorrect classes j != y_i. When 0 is correct class

        G[self.grad_ctx['row_range'], self.grad_ctx['y']] = -1 * torch.sum(G, axis=1) # For j == y_i, build the 1's vector and sum.

        gradient = torch.matmul(self.grad_ctx['x'].T, G)
        gradient = gradient / self.grad_ctx['M'].shape[0]
        # ========================

        return gradient
