# we will implement the entire method from scratch, 
# including (i) the model; (ii) the loss function; (iii) a minibatch stochastic gradient descent optimizer; and (iv) the training function that stitches all of these pieces together.) 

%matplotlib inline
import torch
from d2l import torch as d2l

# MODEL
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

# The resulting forward method is registered in the LinearRegressionScratch class via add_to_class
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return torch.matmul(X, self.w) + self.b # this is just y = mx + b

# LOSS
# Since [updating our model requires taking the gradient of our loss function,] we ought to (define the loss function first.) Here we use the squared loss function in :eqref:eq_mse. 
# In the implementation, we need to transform the true value y into the predicted value's shape y_hat. 
# The result returned by the following method will also have the same shape as y_hat.
# We also return the averaged loss value among all examples in the minibatch.
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return l.mean()


# MINIBATCH SGD
# We define our SGD class, a subclass of d2l.HyperParameters (introduced in :numref:oo-design-utilities), to have a similar API as the built-in SGD optimizer. 
# We update the parameters in the step method. The zero_grad method sets all gradients to 0, which must be run before a backpropagation step.
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

# We next define the configure_optimizers method, which returns an instance of the SGD class.
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    return SGD([self.w, self.b], self.lr) 

# NOTE: .grad computes the gradient

# TRAINING
#Now that we have all of the parts in place (parameters, loss function, model, and optimizer), we are ready to [implement the main training loop.]
# It is crucial that you understand this code fully since you will employ similar training loops for every other deep learning model covered in this book. 
# In each epoch, we iterate through the entire training dataset, passing once through every example (assuming that the number of examples is divisible by the batch size). 
# In each iteration, we grab a minibatch of training examples, and compute its loss through the model's training_step method.
# Then we compute the gradients with respect to each parameter. Finally, we will call the optimization algorithm to update the model parameters. In summary, we will execute the following loop:
# Initialize parameters  (w,b) 
# Repeat until done
  # Compute gradient  g←∂(w,b)1|B|∑i∈Bl(x(i),y(i),w,b) 
  # Update parameters  (w,b)←(w,b)−ηg

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch
  
@d2l.add_to_class(d2l.Trainer)  #@save # This is a decorator used to add the fit_epoch method to the d2l.Trainer class. It seems to be an external library-specific functionality that allows you to extend or modify the behavior of an existing class, d2l.Trainer in this case.
def fit_epoch(self): 
    self.model.train() # This line sets the model into training mode. In PyTorch, when you call .train() on a model, it prepares the model for training, which means that it will keep track of gradients for backpropagation.
    for batch in self.train_dataloader: # This code starts a loop to iterate through batches of training data
        loss = self.model.training_step(self.prepare_batch(batch)) #  For each batch in the training data, it calculates the loss through the model's training_step method. 
        self.optim.zero_grad() # This line clears the gradients of the model's parameters before backpropagation. This step is necessary to ensure that gradients do not accumulate from previous batches.
        with torch.no_grad():  # This context manager is used to temporarily disable gradient tracking for certain operations within the block. This is often used for validation or inference to save memory and processing time since gradients are not needed in these cases.
            loss.backward() # Within the torch.no_grad() block, this line computes the gradients of the loss with respect to the model's parameters. This is the backpropagation step.
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model) # This checks whether gradient clipping is enabled, where self.gradient_clip_val represents the threshold value for gradient clipping. Gradient clipping is a technique to prevent gradients from becoming too large, which can lead to convergence issues in training.
            self.optim.step()  # This line updates the model's parameters using the calculated gradients and the optimization algorithm specified in self.optim.
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval() # This sets the model into evaluation mode. In evaluation mode, the model won't track gradients, and dropout layers (if any) may behave differently.
    for batch in self.val_dataloader: #  If a validation dataloader is provided, this loop iterates through batches of validation data.
        with torch.no_grad():  # Similar to what was done during training, this context manager disables gradient tracking for validation.
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1


# Use the SyntheticRegressionData class and pass in some ground truth parameters. Then we train our model with the learning rate lr=0.03 and set max_epochs=3. 
# Note that in general, both the number of epochs and the learning rate are hyperparameters. 
# In general, setting hyperparameters is tricky and we will usually want to use a three-way split, one set for training, a second for hyperparameter selection, and the third reserved for the final evaluation.
model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)

# Because we synthesized the dataset ourselves, we know precisely what the true parameters are. 
# Thus, we can [evaluate our success in training by comparing the true parameters with those that we learned] through our training loop. Indeed they turn out to be very close to each other.

with torch.no_grad():
    print(f'error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')

# output: error in estimating w: tensor([ 0.1408, -0.1493])
#         error in estimating b: tensor([0.2130])

