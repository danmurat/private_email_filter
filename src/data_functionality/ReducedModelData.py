"""
Holds the PCA reduced training and testing data to be saved and loaded.
"""


class ReducedModelData:
    def __init__(self, red_X_train, red_X_test):
        self.red_X_train = red_X_train
        self.red_X_test = red_X_test

    def get_all_data(self) -> tuple:
        return self.red_X_train, self.red_X_test

    def get_red_X_train(self):
        return self.red_X_train

    def get_red_X_test(self):
        return self.red_X_test
