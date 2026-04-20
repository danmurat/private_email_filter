import tenseal as ts
import torch
from ts_compat_models.LogisticReg import LogisticReg
from ts_compat_models.LinearSVM import LinearSVM

"""
tenseal uses Microsoft's SEAL c++ lib, and gives us a python interface to interact with it.
This is a lot more barebones than zama - so we'll need to write down the algorithms for the
models - applying encrypted addition/multiplication operations when needed.

TRAINING MODELS IN HERE REQUIRE X_train, y_train to be TORCH TENSORS!
"""
class TenSealModels:
    def __init__(self):
        # from tutorial. This holds all the encryption keys + operations we can do
        # we can test CKKS scheme too!
        #self.context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)
        self.c = 200.0 # most accurate soft-margin param found so far for SVM

    def trainLog(self, X_train, y_train, epochs):
        model = LogisticReg(X_train.shape[1])
        optim = torch.optim.SGD(model.parameters(), lr=1) # gradient descent 
        l = torch.nn.BCELoss() # Binary Cross Entropy Loss

        # typical "minimise loss", with pytorch handling most of the details..
        for e in range(epochs + 1):
            optim.zero_grad()
            out = model(X_train)
            loss = l(out, y_train)
            loss.backward()
            optim.step()
            
            if e % 100 == 0: 
                print(f"Loss at epoch {e}: {loss.data}")

        return model

    # def trainLogReg(self, X_train, y_train, epochs):
    #     model = LogisticReg(X_train.shape[1])
    #     model.fit(X_train, y_train, epochs) # this creates a new obj again inside the function. Not sure if it'll break?
    #
    #     return model

    def logAccuracy(self, log_model, X_test, y_test):
        # i know log_model.blahblah(pass in the same object again...). Bear with
        log_model.testAccuracy(log_model, X_test, y_test)
    
    
    # epochs=5000 best we've found with step_size=0.001
    def trainSVM(self, X_train, y_train, epochs=5000):
        model = LinearSVM(X_train.shape[1], self.c)
        model.train(X_train, y_train, epochs)

        return model

    def svmAccuracy(self, ts_svm_model, X_test, y_test):
        ts_svm_model.testAcc(X_test, y_test)







    # DELETE WHEN WE BETTER UNDERSTAND ALL THIS!
    # used just to test tenseal is actually working... IT DOES
    def test(self):
        vector = [1,2,3,4,5]
        enc_vector = ts.bfv_vector(self.context, vector)

        print(f"plain vector: {str(vector)}, and we'll add all by 1")
        print(f"Encrypted vector: {str(enc_vector)}")

        addition_result = enc_vector + [1,1,1,1,1]
        print(f"Result = {str(addition_result._decrypt())}")
