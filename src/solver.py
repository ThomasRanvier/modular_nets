import numpy as np
import pickle as pickle
from copy import deepcopy
from src.optims.adam import Adam

class Solver(object):
    """
    Class that implements a solver, this is mainly just a reimplementation of the 
    solver object from Stanford CS231N course.
    """
    def __init__(self, model, data, **kwargs):
        """
        Instantiates a solver to train the selected model.
        :param model: The model to train.
        :type model: Net.
        :param data: The datas to train and validate with.
        :type data: A dictionary of numpy arrays:
            - 'X_train': Training datas.
            - 'y_train': Training labels.
            - 'X_val': Validation datas.
            - 'y_val': Validation labels.
        :param **kwargs: Optional parameters:
            - optims: The optimisers to use, a list of any optimiser object found 
              under the optims directory.
            - lr_decay: A scalar for learning rate decay; after each epoch the
              learning rate is multiplied by this value.
            - batch_size: Size of minibatches used to compute loss and gradient
              during training.
            - num_epochs: The number of epochs to run for during training.
            - num_train_samples: Number of training samples used to check training
              accuracy; default is 1000; set to None to use entire training set.
            - num_val_samples: Number of validation samples to use to check val
              accuracy; default is None, which uses the entire validation set.
            - checkpoint_name: If not None, then save model checkpoints here every
              epoch.
            - print_every: Integer; training losses will be printed every
              print_every iterations.
            - verbose: Boolean; if set to false then no output will be printed
              during training.
        """
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        #Create a new optimiser object for each connected layer in the model (The
        #first layer not being a connected layer).
        default_optims = []
        for _ in range(len(self.model.layers_sizes) - 1):
            default_optims.append(Adam())
        self.optims = kwargs.pop('optims', default_optims)
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', None)
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        #Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        self.loss_history = []
        self.best_model = None
        self.best_val_acc = 0.0
        self.train_acc_history = []
        self.val_acc_history = []
        self.epoch = 0

    def __step(self):
        """
        Make a single gradient update. This is called by train() and should not be
        called in any other way.
        """
        #Make a minibatch of training data
        num_train = self.X_train.shape[0]
        X_batch = self.X_train
        y_batch = self.y_train
        if num_train > self.batch_size:
            batch_mask = np.random.choice(num_train, self.batch_size)
            X_batch = self.X_train[batch_mask]
            y_batch = self.y_train[batch_mask]
        #Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)
        #Update the parameters.
        i = 0
        for layer in self.model.layers:
            if layer.layer_type == 'connected':
                dw = grads[i]['w']
                layer.weights = self.optims[i].update(layer.weights, dw)
                i += 1

    def __save_checkpoint(self):
        """
        Saves a checkpoint of the trainer with all its parameters in a file.
        """
        if self.checkpoint_name is None: return
        checkpoint = {
          'model': self.model,
          'optims': self.optims,
          'lr_decay': self.lr_decay,
          'batch_size': self.batch_size,
          'num_train_samples': self.num_train_samples,
          'num_val_samples': self.num_val_samples,
          'epoch': self.epoch,
          'loss_history': self.loss_history,
          'train_acc_history': self.train_acc_history,
          'val_acc_history': self.val_acc_history,
        }
        filename = '%s_epoch_%d.pkl' % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        

    def check_accuracy(self, X, y, num_samples = None, batch_size = 100):
        """
        Check the accuracy of the model on the provided datas.
        :param X: The datas to check.
        :type X: A numpy array of shape (N, d_1, ..., d_k).
        :param y: The labels of the datas.
        :type y: Anumpy of shape (N,).
        :param num_samples: If not None, subsample the data and only test the model
        on num_samples datapoints.
        :type num_samples: integer.
        :param batch_size: Split X and y into batches of this size to avoid using
        too much memory.
        :type batch_size: integer.
        """
        #Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]
        #Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)
        return acc

    def predict(self, X, batch_size = 100):
        """
        Predict the result of the model on the provided datas.
        :param X: The datas to check.
        :type X: A numpy array of shape (N, d_1, ..., d_k).
        :param batch_size: Split X and y into batches of this size to avoid using
        too much memory.
        :type batch_size: integer.
        """
        #Compute predictions in batches
        N = X.shape[0]
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        return y_pred

    def train(self):
        """
        Train the model with the given datas.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch
        for t in range(num_iterations):
            self.__step()
            #If in verbose mode, maybe print the training loss.
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (\
                    t + 1, num_iterations, self.loss_history[-1]))
            #Apply lrdecay and increment the epoch counter at the end of each epoch.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for i in range(len(self.optims)):
                    self.optims[i].learning_rate *= self.lr_decay
            #Check train and val accuracy on the first iteration, the last
            #iteration, and at the end of each epoch.
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train,
                    num_samples=self.num_train_samples)
                val_acc = self.check_accuracy(self.X_val, self.y_val,
                    num_samples=self.num_val_samples)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self.__save_checkpoint()
                if self.verbose:
                    print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                           self.epoch, self.num_epochs, train_acc, val_acc))
                #Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_model = deepcopy(self.model)
        self.model = self.best_model
