import spam_imp.src.util as util
import numpy as np
import matplotlib.pyplot as plt
from spam_imp.src.ZamaModels import ZamaModels
from spam_imp.src.data_functionality.PreProcess import PreProcess
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

model_data = util.loadModelPickle(util.model_data_path())
red_model_data = util.loadModelPickle(util.reduced_model_data_path())
X_train, y_train, X_test, y_test = model_data.get_all_data()
red_X_train, red_X_test = red_model_data.get_all_data()

def main():
    n_components = 1400
    pca_accum = pca_scree_test(n_components)
    svd_accum = svd_scree_test(n_components)

    plot_both(n_components, pca_accum, svd_accum)

    # component_blocks = [50, 100, 150, 200, 300]
    # #pca_block_test_loop(component_blocks)
    #
    #
    # # well.. tfidf turning out to be quite a bit better than my own lol
    # svd_block_test_loop(component_blocks, X_train, X_test)
    #
    #
    # # TESTING BoW vs tfidf
    # #pca_block_test_loop(component_blocks, X_train, X_test)
    # #pca_block_test_loop(component_blocks, tfidf_train, tfidf_test)
    # print(f"new bow shape = {X_train.shape}")
    #


    # print(red_X_train.shape)
    # print(text_data.shape)
    #test_og_pca()


def plot_one(name, n, accum):
    components = np.arange(n)
    plt.plot(components, accum)
    plt.title(f"{name} Scree plot")
    plt.xlabel("Component")
    plt.ylabel("Variance explained")
    #plt.axhline(y=0.95, color="red", linestyle="--")
    #plt.savefig("screeplot_large.png")
    plt.show()


def plot_both(n, pca_accum, svd_accum):
    components = np.arange(n)
    # plt.figure(figsize=(8,5))
    # plt.step(range(1, len(accum) + 1), accum, where="mid")
    plt.plot(components, pca_accum, color="blue")
    plt.plot(components, svd_accum, color="green")
    plt.title("PCA (blue) and SVD (green) Scree plot")
    plt.xlabel("Component")
    plt.ylabel("Cumulative Variance explained")
    #plt.axhline(y=0.80, color="red", linestyle="--")
    #plt.savefig("screeplot_large.png")
    plt.show()


def pca_scree_test(n):
    # just testing 800 to see how the variance changes (perhaps we'd need more than 200!)
    pca = PCA(n_components=n)

    red_data = pca.fit(X_train)

    component_variances = pca.explained_variance_ratio_
    accum = np.cumsum(component_variances)

    return accum

def svd_scree_test(n):
    # just testing 800 to see how the variance changes (perhaps we'd need more than 200!)
    svd = TruncatedSVD(n_components=n)

    red_data = svd.fit(X_train)

    component_variances = svd.explained_variance_ratio_
    accum = np.cumsum(component_variances)

    return accum

def pca_block_test_loop(component_blocks, train, test):
    for c in component_blocks:
        print("Training with block: ", c)
        pca = PCA(n_components=c)
        red_X_train = pca.fit_transform(train)
        red_X_test = pca.transform(test)
        train_and_test_block(red_X_train, red_X_test)
        print(f"Block: {c} complete.\n\n")

def svd_block_test_loop(component_blocks, train, test):
    for c in component_blocks:
        print("\nTraining with block: ", c)
        svd = TruncatedSVD(n_components=c, algorithm="arpack")
        svd_red_X_train = svd.fit_transform(train)
        svd_red_X_test = svd.transform(test)
        train_and_test_block(svd_red_X_train, svd_red_X_test)

def train_and_test_block(train, test):
    z = ZamaModels()

    print("training svm model...")
    svm = z.trainSVM(train, y_train)

    print("training complete!\n")

    print("training logistic regresssion model...")
    log = z.trainLogistic(train, y_train)
    print("training complete!\n")

    # quick accuracy test to make sure we didn't f up
    print("zama plain logistic accuracy: ")
    z.testPlainAccuracy(log, test, y_test)
    print("zama plain svm accuracy: ")
    z.testPlainAccuracy(svm, test, y_test)


def test_og_pca():
    print("testing og pca model (200 components)")
    z = ZamaModels()


    pca = PCA(n_components=200)
    # newpca_X_train = pca.fit_transform(X_train)
    # newpca_X_test = pca.fit_transform(X_test)
    # I think when I pca'd a few months back, the last fit_transform was on the test set.
    # When I load the pca object to transform my text, it's likely i'm transforming everything based off
    # the test set fit!!
    # below will check to confirm (if it has same acc as loaded pca)
    pca.fit(X_train)
    newpca_X_train = pca.transform(X_train)
    newpca_X_test = pca.transform(X_test)
    # that's exactly what i did... yikes...


    print(f"Og pca train shape = {red_X_train.shape}")
    print(f"New pca train shape = {newpca_X_train.shape}")

    print("training svm models...")
    ogpca_svm = z.trainSVM(red_X_train, y_train)
    newpca_svm = z.trainSVM(newpca_X_train, y_train)

    print("training complete!\n")

    print("training logistic regresssion models...")
    ogpca_log = z.trainLogistic(red_X_train, y_train)
    newpca_log = z.trainSVM(newpca_X_train, y_train)
    print("training complete!\n")

    # quick accuracy test to make sure we didn't f up
    print("og pca plain svm accuracy: ")
    z.testPlainAccuracy(ogpca_svm, red_X_test, y_test)
    print("new pca plain svm accuracy: ")
    z.testPlainAccuracy(newpca_svm, newpca_X_test, y_test)
    print("og pca logistic accuracy: ")
    z.testPlainAccuracy(ogpca_log, red_X_test, y_test)
    print("new pca plain logistic accuracy: ")
    z.testPlainAccuracy(newpca_log, newpca_X_test, y_test)


if __name__ == "__main__":
    main()