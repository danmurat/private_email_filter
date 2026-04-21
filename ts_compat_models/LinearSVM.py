import torch
import numpy as np
import tenseal as ts

"""
Formulas from: https://en.wikipedia.org/wiki/Support_vector_machine
pytorch used here to apply gradient descent on weights and bias
Finally works now! C must be big enough (> 100.0); likely to do with the vastnass of our data (thousands of dimensions very likely overlapping a lot)
97% accuracy on training data with C=200.0-800.0
"""
class LinearSVM:
    def __init__(self, n_dimensions, c):
        self.n_dimensions = n_dimensions
        # requires_grad let's us easily update w's/b when minimising loss
        self.w = torch.rand(n_dimensions, requires_grad=True)
        self.b = torch.rand(1, requires_grad=True)
        self.step_size = 0.001
        self.c = c

    # so we can predict outside the class?
    def get_w(self):
        return self.w.detach().numpy() # requires_grad might be bloating/complicating things (detach removes)

    def get_b(self):
        return self.b.detach().numpy()

    # TODO: figure out why our accuracy is staying at 50%... 
    # apr 7th: turns out that having [0,1] as labels (specifically 0) fucks up the loss function
    # if y_i == 0, 1 - 0 * (whatever) = 1, max(0,1) = 1 and nothing is learnt for the other end
    # because we never really get a negative number?? but this would pick 0 since it's higher?
    # I don't completely understand, but I'll change 0's to -1's to see if this sorts it.
    # FIXED: self.c wasn't large enough. Dataset is more mixed, so large c needed (> 100.0)
    def train(self, X, y, epochs):
        y = torch.where(y <= 0.0, torch.tensor(-1.0), torch.tensor(1.0))
        y = y.squeeze()
        prev_loss = [] # acts like a stack so we can see how much difference to the previous one we had
        prev_params = []
        for e in range(epochs + 1): # +1 so we can print the final 100th epoch
            m = torch.matmul(X, self.w) - self.b

            #print(f"shape of y = {y.shape} | shape of m = {m.shape}")
            l = torch.relu(1 - y * m).mean()

            # apparantly 0.5 is a good regularisation constant for this?? 
            w_reg = 0.5 * torch.sum(self.w**2)
            loss = w_reg + self.c * l
            
            loss.backward()

            with torch.no_grad():
                self.w -= self.step_size * self.w.grad
                self.b -= self.step_size * self.b.grad

            self.w.grad.zero_()
            self.b.grad.zero_()


            if e % 100 == 0: 
                print(f"Loss at epoch {e}: {loss.item()}")

        print(f"Final loss at epoch {e}: {loss.item()}")
    
    
    def testAcc(self, X, y):
        correct_counter = 0
        length = len(X)
        # convert y tensor to integer
        y_compare = torch.tensor(y).int().squeeze()
        y_compare = y_compare.tolist()
        y_pred_list = []


        for i in range(length):
            # sign just turns anything positive to 1 and negative to -1
            # item gets rid of the computational graph we create when using "grad"
            y_pred = int(torch.sign(torch.dot(self.w, X[i]) - self.b).item())
            if y_pred == -1: 
                y_pred = 0
            y_pred_list.append(y_pred)
            if y_pred == y[i]:
                correct_counter += 1

        accuracy = correct_counter / length
        print(f"SVM Accuracy = {accuracy}")
        #print(y_pred_list)


    # predict and enc_predict for actually guessing a given x value

    def predict(self, x_i):
        w = self.w.detach().numpy()
        b = self.b.detach().numpy()
        y = int(np.sign(np.dot(w, x_i) - b))
        if y == -1: y = 0

        return y

    # i think for encrypted, I'd have to do it in TenSealModels, where the encryption context is?
    # actually, a new class that holds this object + handles encryption stuff

    # The model is hypothetically on a server, so methods here are server methods
    # we'll be given an enc_x which should have the .dot associated with it (from whatever key ops happened)
    def enc_prelim_predict(self, enc_x_i):
        # also, not sure if w/b have to be encrypted too? Shouldn't be.. (you can add plaintext 1's to encrypted vector)
        w = self.w.detach().numpy() # interpreter complaining about requires_grad
        b = self.b.detach().numpy()

        return enc_x_i.dot(w) - b
