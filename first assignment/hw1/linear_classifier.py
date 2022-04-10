import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss



class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        # ====== YOUR CODE: ======
        torch_norm_dist = torch.distributions.normal.Normal(loc=0, scale=weight_std)
        self.weights = torch_norm_dist.sample((n_features, n_classes))
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        # ====== YOUR CODE: ======
        class_scores = torch.matmul(x, self.weights)
        max, y_pred = torch.max(class_scores, dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        # ====== YOUR CODE: ======
        accuracy = torch.isclose(y, y_pred).sum().item() / y.shape[0]
        # ========================

        return accuracy * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            accuracy_list = []
            accumulated_loss = 0
            for x_train, y_train in dl_train:
                y_pred, class_scores = self.predict(x_train)
                loss = loss_fn.loss(x_train, y_train, class_scores, y_pred)
                gradient = loss_fn.grad()
                regularization = weight_decay * self.weights
                self.weights = self.weights - learn_rate * (gradient + regularization) #update weights

                loss = loss + (weight_decay/2) * ( torch.norm(self.weights) ** 2)
                accumulated_loss += loss #accumulating per mini-batch
                accuracy_list.append( self.evaluate_accuracy(y_train, y_pred) ) # saving the accuracy of the mini-batch

            train_res.accuracy.append(sum(accuracy_list) / len(accuracy_list)) #calculate accuracy for the epoch
            train_res.loss.append( accumulated_loss / len(accuracy_list) ) #calculate average loss of all mini-batches for the epoch

            accuracy_list = []
            accumulated_loss = 0
            for x_validation, y_validation in dl_valid:
                y_pred, class_scores = self.predict(x_validation)
                loss = loss_fn.loss(x_validation, y_validation, class_scores, y_pred)
                #No weight update in validation
                loss = loss + (weight_decay / 2) * (torch.norm(self.weights) ** 2)
                accumulated_loss += loss  # accumulating per mini-batch
                accuracy_list.append(self.evaluate_accuracy(y_validation, y_pred))

            valid_res.accuracy.append(sum(accuracy_list) / len(accuracy_list))
            valid_res.loss.append( accumulated_loss / len(accuracy_list))



            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        w_images = self.weights
        C, H, W = img_shape[0], img_shape[1], img_shape[2]
        if has_bias == True:
            w_images = w_images[1:,:]

        w_images = w_images.T.reshape( self.n_classes, C, H, W)
        # ========================
        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp['weight_std'] = 0.01
    hp['learn_rate'] = 0.01
    hp['weight_decay'] = 0.001
    # ========================

    return hp
