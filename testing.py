from TenSealModels import TenSealModels
from PreProcess import PreProcess
from HandleModel import HandleModel
from tenseal_models.LinearSVM import LinearSVM
import torch

"""
Any random testing needed is done in here.
"""

p = PreProcess()
p.preprocess()
h = HandleModel(p)
h.dataSetup()

X_train, y_train, X_test, y_test = h.getData()
# convert data to torch tensors
t_X_train = torch.tensor(X_train).float()
t_y_train = torch.tensor(y_train).float().unsqueeze(1) # adds extra dimension [n, 1] so that it corresponds with X when training
t_X_test = torch.tensor(X_test).float()
t_y_test = torch.tensor(y_test).float().unsqueeze(1)

dimensions = X_train.shape[1]

t = TenSealModels()

def testLR():
    lr = t.trainLogReg(t_X_train, t_y_train, 30)

    t.testPlainLogReg(lr, t_X_test, t_y_test)

# C at 200 - 800 pretty much all get 97% accuracy
def testSVM():
    svm0 = LinearSVM(dimensions, 200.0)
    # print("training svm with c=200.0")
    # svm0.train(t_X_train, t_y_train, 5000)
    # print("training complete!\n")
    # svm1 = LinearSVM(dimensions, 300.0)
    # print("training svm with c=300.0")
    # svm1.train(t_X_train, t_y_train, 5000)
    # print("training complete!\n")
    # svm2 = LinearSVM(dimensions, 700.0)
    # print("training svm with c=700.0")
    # svm2.train(t_X_train, t_y_train, 5000)


    svm0.testAcc(t_X_test, t_y_test) 
    # svm1.testAcc(t_X_test, t_y_test)
    # svm2.testAcc(t_X_test, t_y_test)

def testTenSeal():
    t.test()

if __name__ == "__main__":
    #testLR()
    testSVM()
