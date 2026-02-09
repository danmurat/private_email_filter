from PreProcess import PreProcess
from HandleModel import HandleModel
import time
import random
import numpy as np
import math
import scipy.stats as st
import sys

p = PreProcess()
p.preprocess()
h = HandleModel(p)
# h.dataSetup # uncomment this if we're just training a new one



# not sure whether to just train a new one? Or load the saved model and do all the encrypt/decrypt shinanigans
svm_model = h.loadModel("svm")
svm_model.load()

log_model = h.loadModel("log")
log_model.load()

# we need access to the test data. ALL of it
def main():
    y_test = p.getTestingData()["label"]
    X_test = p.getVectorisedTestText()
    sample_size = 110 # we'll drop the first 10 inferences (incase of cold starts)
    X_rand, y_rand = randomise(sample_size, X_test, y_test)

    runModelTesting(X_rand, "svm") # 9th feb benchmarking Plain! not HE!
    runModelTesting(X_rand, "log")

def runModelTesting(rand_texts, model_name):
    print(f"Testing {model_name}")

    inference_times = benchmarkModel(rand_texts, model_name)

    print(inference_times)
    
    mean, sd, var, n = getStats(inference_times)
    printStats(mean, sd, var, n)
    err = calcErrorMargin(sd, n)

    mean_low, mean_high = getMeanRange(mean, err)

    print(f"Inference mean: {mean_low}ms-{mean_high}ms, 95% CI")



# prints stuff like mean, sd, variance, etc...
def getStats(data) -> tuple:
    mean = np.mean(data)
    sd = np.std(data, ddof=1)
    var = np.var(data, ddof=1)
    n = len(data)

    return (mean, sd, var, n)

def printStats(mean, sd, var, n):
    print("Data size = ", n)
    print("Mean = ", mean)
    print("Standard Deviation = ", sd)
    print("Variance = ", var)


def calcErrorMargin(sd, n):
    z = 1.96 # this is our 95% confidence interval
    e = z * (sd / math.sqrt(n))
    
    return e

# gives us our mean range (from confidence interval)
def getMeanRange(mean, err) -> tuple:
    low = mean - err
    high = mean + err

    return (low, high)



# returns n timed inferences
def benchmarkModel(X_rand, model_name) -> list:
    cli, eval_keys = h.client(model_name)

    inferenceTimes = []

    model = None
    pca_model = None
    if model_name == "svm":
        model = h.loadModel("svm")
    elif model_name == "log":
        model = h.loadModel("log")
    elif model_name == "pca_svm":
        model = h.loadModel("pca_svm")
        pca_model = h.loadModelPickle("pca")
    elif model_name == "pca_log":
        model = h.loadModel("pca_log")
        pca_model = h.loadModelPickle("pca")
    else:
        print(f"INCORRECT MODEL SELECTION! (svm) or (log) or (pca_ in front)! Your's was {model_name}")
        sys.exit()

    model.load()

    for i in range(len(X_rand)):
        enc_data = None 
        if pca_model == None:
            enc_data = cli.quantize_encrypt_serialize(X_rand[i:i+1]) # i:i+1 preserves (1,n) 2d-ness
        else:
            data = pcaReduceEmail(pca_model, X_rand[i:i+1])
            enc_data = cli.quantize_encrypt_serialize(data)
        inferenceTimes.append(inferenceTime(model, enc_data, eval_keys))
        print(f"Inference {i} complete.")

    return inferenceTimes[10:] # dropping first 10 elements

def benchmarkModelPlain(X_rand, model_name):
    inferenceTimes = []
    model = None
    pca_model = None
    
    if model_name == "svm":
        model = h.loadModelPickle("svm")
    elif model_name == "log":
        model = h.loadModelPickle("log")
    elif model_name == "pca_svm":
        model = h.loadModelPickle("pca_svm")
        pca_model = h.loadModelPickle("pca")
    elif model_name == "pca_log":
        model = h.loadModelPickle("pca_log")
        pca_model = h.loadModelPickle("pca")
    else:
        print(f"INCORRECT MODEL SELECTION! (svm) or (log) or (pca_ in front)! Your's was {model_name}")
        sys.exit()

    for i in range(len(X_rand)):
        data = None 
        if pca_model == None:
            data = X_rand[i:i+1] # i:i+1 preserves (1,n) 2d-ness
        else:
            data = pcaReduceEmail(pca_model, X_rand[i:i+1])
        inferenceTimes.append(inferenceTimePlain(model, data))
        print(f"Inference {i} complete.")

    return inferenceTimes[10:] # dropping first 10 elements
 

def pcaReduceEmail(pca, vectorised_email):
    return pca.transform(vectorised_email)

# selects random index's for us to test (n=30 for a pilot test)
def randomise(n, X, y) -> tuple:
    # these might have to be changed to an np.array
    randomised_X = np.zeros((n, len(X[0])))
    randomised_y = np.zeros((n, 1)) 
    for i in range(n):
        rand = random.randint(0, 2000)
        randomised_X[i] = X[rand]
        randomised_y[i] = y[rand]

    return (randomised_X, randomised_y)


def inferenceTime(model, enc_data, eval_keys) -> float:
    start = time.perf_counter()
    model.run(enc_data, eval_keys) # we don't care about the result for now
    end = time.perf_counter()             # but might have to do due diligence and check!
    return end - start

def inferenceTimePlain(model, data) -> float:
    start = time.perf_counter()
    model.predict(data)
    end = time.perf_counter()
    return end - start



if __name__ == "__main__":
    main()