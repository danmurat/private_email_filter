from sympy import false

from ZamaModels import ZamaModels
import util
import time
import random
import numpy as np
import math
import scipy.stats as st
import tenseal as ts
import client

# selects random index's for us to test (n=30 for a pilot test)
def randomise(n, X, y) -> tuple:
    # these might have to be changed to an np.array
    randomised_X = np.zeros((n, len(X[0])))
    randomised_y = np.zeros((n, 1))
    for i in range(n):
        rand = random.randint(0, 1999) # 2000 is out of bounds..
        randomised_X[i] = X[rand]
        randomised_y[i] = y[rand]

    return randomised_X, randomised_y

z = ZamaModels() # just to use the loadModels() method 

# these as global variables, so we can call as needed (without having to pass params)
model_data = util.loadModelPickle(util.model_data_path())
reduced_model_data = util.loadModelPickle(util.reduced_model_data_path())

X_test = model_data.get_X_test() # MAKE SURE TO NOT INCLUDE THESE WHEN TESTING, BY ACCIDENT!!! (yes i spent the past hr debugging this)
y_test = model_data.get_y_test()
red_X_test = reduced_model_data.get_red_X_test()

sample_size = 610 # we'll drop the first 10 inferences (incase of cold starts)
X_rand, y_rand = randomise(sample_size, X_test, y_test)
red_X_rand, red_y_rand = randomise(sample_size, red_X_test, y_test)


### ---------------- ###

def main():
    #run_model_testing("zama_svm") # 9th feb benchmarking Plain! not HE!
    #run_model_testing("zama_log") # working with refactors
    #run_model_testing("zama_pca_log") # but does it work pca? It does :)
    #run_model_testing("zama_pca_svm")

    #test_model_accuracy("ts_svm") # 9th apr testing

    # TODO: 10th apr
    # - handle inference testing for tenseal svm [done]
    # - get encrypted inference implemented for tenseal log [in progress]
    # - make inference testing work for both tenseal svm/log

    #run_model_testing("ts_svm") # 10th apr testing 70ms! vs 340ms for zama..

    #test_model_accuracy("ts_svm") # HE accuracy 95%+ for log and svm

    #run_model_testing("ts_log") # inference also ~69/70ms (key size: 2^14)
                                # inference at 24ms if key size = 2^13 rather than
                                # but input doesn't fit into 2^13, so should theoretically have some accuracy drop (it's like 1-0.5% drop when testing)


### ---------------- ###

def run_model_testing(model_name):
    print(f"Testing {model_name}")

    inference_times = None
    if model_name[0] == 'z':
        inference_times = zama_benchmark_model(model_name[5:]) # zama_log -> log
    elif model_name[0] == 't':
        inference_times = ts_benchmark_model(model_name[3:])
    else:
        raise ValueError(f"model_name `{model_name}` should begin with `zama_` or `ts_`")

    print(inference_times)
    
    mean, sd, var, n = getStats(inference_times)
    printStats(mean, sd, var, n)
    err = calcErrorMargin(sd, n)

    mean_low, mean_high = getMeanRange(mean, err)

    print(f"Inference mean: {mean_low}ms-{mean_high}ms, 95% CI")


def test_model_accuracy(model_name):
    accuracy = None
    if model_name[0] == 'z':
        accuracy = zama_test_accuracy(model_name[5:]) # zama_log -> log
    elif model_name[0] == 't':
        accuracy = ts_test_accuracy(model_name[3:])
    else:
        raise ValueError(f"model_name `{model_name}` should begin with `zama_` or `ts_`")


# potentially return accuracy values (instead of just printing?). Don't see why I need to do this? Just show once
# how accurate it is, and save it on paper...
def ts_test_accuracy(model_name):
    model = util.loadModelPickle(f"ts_plain_models/{model_name}")
    ctx_eval = client.setup_ts_params() # remember to change depending on what the model is (when we test log)

    isLog = True
    if model_name == "svm": # potentially make more robust. What if it's neither?
        isLog = False


    print("Encrypting test set...")
    enc_x = client.ts_encrypt_x(X_rand, ctx_eval)
    print("Test set encrypted.")

    total = len(y_rand)
    correct_counter = 0

    print("Testing accuracy...")

    for i in range(sample_size):
        enc_prelim_y = model.enc_prelim_predict(enc_x[i])
        prelim_y = enc_prelim_y.decrypt()
        pred_y = None
        if isLog:
            pred_y = client.ts_client_finish_prediction_log(prelim_y)
            if client.ts_is_correct_prediction_log(y_rand[i], pred_y):
                correct_counter += 1
        else:
            pred_y = client.ts_client_finish_prediction_svm(prelim_y)
            if client.ts_is_correct_prediction_svm(y_rand[i], pred_y):
                correct_counter += 1

    print(f"\nTENSEAL {model_name} HE ACCURACY = {correct_counter / total}%")


# call incase the need to debug
def ts_test_plain_accuracy(model_name):
    model = util.loadModelPickle(f"ts_plain_models/{model_name}")

    isLog = True
    if model_name == "svm": # potentially make more robust. What if it's neither?
        isLog = False

    total = len(y_rand)
    correct_counter = 0

    print("Testing accuracy...")

    for i in range(sample_size):
        pred_y = model.predict(X_rand[i])
        print(f"Actual y = {y_rand[i]} | predicted y = {pred_y}")
        if pred_y == y_rand[i]:
            correct_counter += 1

    print(f"\nTENSEAL {model_name} HE ACCURACY = {correct_counter / total}%")

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
def zama_benchmark_model(model_name) -> list:
    model = _load_zama_model(model_name)
    model.load()

    inference_times = _zama_inference_test_loop(model, model_name)

    return inference_times


# remove svm name after once we have log set up too (just to remind us we're solely testing svm)
def ts_benchmark_model(model_name):
    model = util.loadModelPickle(f"ts_plain_models/{model_name}")

    enc_x = client.ts_encrypt_x(X_rand, client.setup_ts_params())
    # deal with reduced_x at some point!

    inference_times = _ts_inference_test_loop(model, enc_x)

    return inference_times

def benchmarkModelPlain(model_name):
    inferenceTimes = []
    model = _load_zama_model(model_name)
    pca_model = None


    for i in range(len(X_rand)):
        data = None 
        if pca_model == None:
            data = X_rand[i:i+1] # i:i+1 preserves (1,n) 2d-ness
        else:
            data = pcaReduceEmail(pca_model, X_rand[i:i+1])
        inferenceTimes.append(inferenceTimePlain(model, data))
        print(f"Inference {i} complete.")

    return inferenceTimes[10:] # dropping first 10 elements
 

# TODO: change so that we just get red_X from ReducedModelData (load this)
def pcaReduceEmail(pca, vectorised_email):
    return pca.transform(vectorised_email)



def zama_inference_time(model, enc_x, eval_keys) -> float:
    start = time.perf_counter()
    model.run(enc_x, eval_keys) # we don't care about the result for now
    end = time.perf_counter()             # but might have to do due diligence and check!
    return end - start

# same idea here (appending svm to the name, till we do log too)
def ts_inference_time_svm(model, enc_x_i) -> float:
    start = time.perf_counter()
    enc_prelim_y = model.enc_prelim_predict(enc_x_i) # client does the decrypt and determining sign, so this is all that's needed
    end = time.perf_counter()
    return end - start


def inferenceTimePlain(model, data) -> float:
    start = time.perf_counter()
    model.predict(data)
    end = time.perf_counter()
    return end - start


# _name() functions here (like private) intended to be used within other functions.

def _load_zama_model(model_name):
    model = None
    #pca_model = None # TODO: deal with PCA testing cases
    if model_name == "svm":
        model = z.loadModel("svm")
    elif model_name == "log":
        model = z.loadModel("log")
    elif model_name == "pca_svm":
        model = z.loadModel("pca_svm")
        #pca_model = util.loadModelPickle("pca")
    elif model_name == "pca_log":
        model = z.loadModel("pca_log")
        #pca_model = util.loadModelPickle("pca") # TODO: fix pca'ing. Just get reduced_X data
    else:
        raise ValueError(f"INCORRECT MODEL SELECTION! (svm) or (log) or (pca_ in front)! Your's was {model_name}")

    return model


def _zama_inference_test_loop(model, model_name):
    cli, eval_keys = z.client(model_name)
    inference_times = []
    need_red_x = False

    if model_name[:3] == "pca":
        need_red_x = True

    for i in range(sample_size):
        enc_x = None
        if not need_red_x:
            enc_x = cli.quantize_encrypt_serialize(X_rand[i:i+1]) # i:i+1 preserves (1,n) 2d-ness
        else:
            enc_x = cli.quantize_encrypt_serialize(red_X_rand[i:i+1])
        inference_times.append(zama_inference_time(model, enc_x, eval_keys))
        print(f"Inference {i} complete.")

    return inference_times[10:]

def _ts_inference_test_loop(model, enc_x):
    inference_times = []
    for i in range(sample_size):
        start = time.perf_counter()
        model.enc_prelim_predict(enc_x[i]) # encrypted inference (don't need result)
        stop = time.perf_counter()
        inference_times.append(stop - start)
        print(f"Inference {i} complete.")

    return inference_times[10:]

if __name__ == "__main__":
    main()