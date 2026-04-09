import torch

# https://github.com/OpenMined/TenSEAL/blob/main/tutorials%2FTutorial%201%20-%20Training%20and%20Evaluation%20of%20Logistic%20Regression%20on%20Encrypted%20Data.ipynb
class LogisticReg(torch.nn.Module):
    def __init__(self, n_features):
        # think this is needed to inherit from pytorch's stuff..
        super(LogisticReg, self).__init__()
        # just input/output: data features, and a final decision (spam/ham)
        self.log_reg = torch.nn.Linear(n_features, 1)


    # forward prop for training 
    def forward(self, x):
        return torch.sigmoid(self.log_reg(x))


    # creating the same object (LogisticReg) within this object seems to break things (accuracy mysteriously drops to 50s)
    # using trainLog_test() in TenSealModels for now..
    # def fit(self, X_train, y_train, epochs):
    #     model = LogisticReg(X_train.shape[1])
    #     optim = torch.optim.SGD(model.parameters(), lr=1) # gradient descent 
    #     criterion = torch.nn.BCELoss() # Binary Cross Entropy Loss

    #     # typical "minimise loss", with pytorch handling most of the details..
    #     for e in range(epochs):
    #         optim.zero_grad()
    #         out = model(X_train)
    #         loss = criterion(out, y_train)
    #         loss.backward()
    #         optim.step()
    #         print(f"Loss at epoch {e}: {loss.data}")

    #     return model

    def _accuracy(self, pred, actual):
        correct = torch.abs(actual - pred) < 0.5
        return correct.float().mean()

    def testAccuracy(self, model, X_test, y_test):
        y_pred = model(X_test)
        accuracy = self._accuracy(y_pred, y_test)

        print(f"Logistic reg acc = {accuracy}") 