import numpy as np

class Loss_computer():
    """
    Class that implements a loss computer, used to compute the loss from the score 
    and labels.
    The default loss method is softmax.
    """
    def __init__(self, method = 'softmax'):
        """
        Instantiates a loss computer, an object that applies the selected loss.
        :param method: The chosen loss method.
        :type method: str.
        """
        if not hasattr(self, method):
            raise ValueError('Invalid loss method: ' + method)
        self.method = getattr(self, method)

    def compute_loss(self, x, y):
        """
        Compute the loss by applying the defined loss function.
        :param x: The scores to apply the loss to.
        :type x: A numpy array of shape (N, D).
        :param y: The labels corresponding to the datas.
        :type y: A numpy array of shape (N,)
        :return (loss, dx): The loss and gradient of the loss with respect of x.
        :rtype (loss, dx): A tuple of a float and a numpy array.
        """
        return self.method(x, y)

    def svm(self, x, y):
        """
        Computes the loss and gradient with SVM method.
        :param x: The scores to apply the loss to.
        :type x: A numpy array of shape (N, D).
        :param y: The labels corresponding to the datas.
        :type y: A numpy array of shape (N,)
        :return (loss, dx): The loss and gradient of the loss with respect of x.
        :rtype (loss, dx): A tuple of a float and a numpy array.
        """
        N = x.shape[0]
        correct_class_scores = x[np.arange(N), y]
        margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
        margins[np.arange(N), y] = 0
        loss = np.sum(margins) / N
        num_pos = np.sum(margins > 0, axis=1)
        dx = np.zeros_like(x)
        dx[margins > 0] = 1
        dx[np.arange(N), y] -= num_pos
        dx /= N
        return loss, dx


    def softmax(self, x, y):
        """
        Computes the loss and gradient with softmax method.
        :param x: The scores to apply the loss to.
        :type x: A numpy array of shape (N, D).
        :param y: The labels corresponding to the datas.
        :type y: A numpy array of shape (N,)
        :return (loss, dx): The loss and gradient of the loss with respect of x.
        :rtype (loss, dx): A tuple of a float and a numpy array.
        """
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = x.shape[0]
        loss = -np.sum(log_probs[np.arange(N), y]) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx
