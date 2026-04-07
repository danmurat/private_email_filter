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