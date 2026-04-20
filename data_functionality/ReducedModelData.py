"""
Holds the PCA reduced training and testing data to be saved and loaded.
"""

class ReducedModelData:
    def __init__(self, pca_red_X_train, pca_red_X_test, svd_red_X_train, svd_red_X_test):
        self.pca_red_X_train = pca_red_X_train
        self.pca_red_X_test = pca_red_X_test
        self.svd_red_X_train = svd_red_X_train
        self.svd_red_X_test = svd_red_X_test

    def get_all_pca_data(self) -> tuple:
        return self.pca_red_X_train, self.pca_red_X_test

    def get_all_svd_data(self) -> tuple:
        return self.svd_red_X_train, self.svd_red_X_test
        
    def get_pca_red_X_train(self):
        return self.pca_red_X_train

    def get_pca_red_X_test(self):
        return self.pca_red_X_test
