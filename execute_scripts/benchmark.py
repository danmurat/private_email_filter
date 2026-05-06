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
import tracemalloc
import psutil
import os
z = ZamaModels() # just to use the loadModels() method

X_test, red_X_test, y_test = util.load_test_data()

sample_size =33 # we'll drop the first 3 inferences (incase of cold starts)
X_rand, y_rand = util.randomise(sample_size, X_test, y_test)
red_X_rand, red_y_rand = util.randomise(sample_size, red_X_test, y_test)


### ---------------- ###

def main():
    #test_model_latency("zama_svm") # 9th feb benchmarking Plain! not HE!
    #test_model_latency("zama_log") # working with refactors
    #test_model_latency("zama_pca_log") # but does it work pca? It does :)
    #test_model_latency("zama_pca_svm")

    # 21st apr:
    # want to test if enc accuracy = plain accuracy. It should be, since HE guarantees the same results.
    # Since we have to "pilot" our accuracy tests too (too long to test whole test set), we can just see
    # if the plaintext accuracy on the same random set is the same
    # test_model_accuracy("ts_svm", True)
    # print("\nTESTING PLAIN\n")
    # test_model_accuracy("ts_svm", False)
    """
    Re-running MULTIPLE times continuously yields the same correct counters for both. Safe to say that the encrypted models
    have the same accuracy as plain, so we'll use the plaintext accuracies on the whole test set when talking about the
    model as a whole. 
    """

    #test_model_latency("plain_ts_svm")

    # 22nd apr
    #test_enc_dec_times("pal_svd_log")

    #test_optimised_ts_server_usage()
    cpu_usage()

### ---------------- ###

def test_model_latency(model_name):
    print(f"Testing {model_name}")

    inference_times = None
    if model_name[:5] != "plain":
        if model_name[0] == 'z':
            inference_times = zama_benchmark_model(model_name[5:]) # zama_log -> log
        elif model_name[0] == 't':
            inference_times = ts_benchmark_model(model_name[3:])
        elif model_name[0] == 'p':
            inference_times = pal_benchmark_model(model_name[4:])
        else:
            raise ValueError(f"model_name `{model_name}` should begin with `zama_` or `ts_`")
    else:
        # we test the plain inference for zama or ts
        model_name = model_name[6:]
        print(model_name)
        if model_name[0] == 'z':
            inference_times = zama_bench_plain_inference(model_name[5:])
        elif model_name[0] == 't':
            inference_times = ts_bench_plain_inference(model_name[3:])

    print(inference_times)
    
    mean, sd, var, n = getStats(inference_times)
    printStats(mean, sd, var, n)
    err = calcErrorMargin(sd, n)

    mean_low, mean_high = getMeanRange(mean, err)

    print(f"Inference mean: {mean_low}ms-{mean_high}ms, 95% CI")

# cpu and mem
def test_optimised_ts_server_usage():
    ctx = client.setup_ts_params()
    ts_svd_svm = util.loadModelPickle("ts_plain_models/svd_svm")

    results = []

    for i in range(sample_size):
        enc_x_i = client.ts_encrypt_x_i(red_X_rand[i], ctx)

        enc_email_bytes = enc_x_i.serialize()
        pub_ctx_bytes = ctx.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_relin_keys=True,
            save_galois_keys=True
        )

        # 'server running'
        tracemalloc.start()
        # process = psutil.Process(os.getpid())
        # process.cpu_percent(interval=None)
        pub_ctx_bytes = pub_ctx_bytes.read()     # should be python bin objs now
        enc_email_bytes = enc_email_bytes.read()

        pub_ctx = ts.context_from(pub_ctx_bytes)
        enc_email = ts.ckks_vector_from(pub_ctx, enc_email_bytes)

        enc_prelim_result = ts_svd_svm.enc_prelim_predict(enc_email)
        enc_presult_bytes = enc_prelim_result.serialize()

        current, peak = tracemalloc.get_traced_memory()
        peak_MB = peak / 10**6
        tracemalloc.stop()
        #cpu_usage = process.cpu_percent(interval=None)

        results.append(peak_MB)
        #results.append(cpu_usage)

    adjusted = results[3:]

    mean, sd, var, n = getStats(adjusted)
    printStats(mean, sd, var, n)
    err = calcErrorMargin(sd, n)

    mean_low, mean_high = getMeanRange(mean, err)

    print(f"Usage: {mean_low}-{mean_high}, 95% CI")


def test_enc_dec_times(model_name):
    print(f"Testing {model_name}")

    enc_times, dec_times = None, None
    if model_name[0] == 'z':
        enc_times, dec_times = zama_benchmark_enc_dec(model_name[5:]) # zama_log -> log
    elif model_name[0] == 't':
        enc_times, dec_times = ts_benchmark_enc_dec(model_name[3:])
    elif model_name[0] == 'p':
        enc_times, dec_times = pal_benchmark_enc_dec(model_name[4:])
    else:
        raise ValueError(f"model_name `{model_name}` should begin with `zama_` or `ts_`")

    print(enc_times)
    # ENCRYPTION STATS
    enc_mean, enc_sd, enc_var, enc_n = getStats(enc_times)
    printStats(enc_mean, enc_sd, enc_var, enc_n)
    enc_err = calcErrorMargin(enc_sd, enc_n)

    enc_mean_low, enc_mean_high = getMeanRange(enc_mean, enc_err)

    print(f"{model_name} Encryption mean: {enc_mean_low}ms-{enc_mean_high}ms, 95% CI\n")

    print(dec_times)
    # DECRYPTION STATS
    dec_mean, dec_sd, dec_var, dec_n = getStats(dec_times)
    printStats(dec_mean, dec_sd, dec_var, dec_n)
    dec_err = calcErrorMargin(dec_sd, dec_n)

    dec_mean_low, dec_mean_high = getMeanRange(dec_mean, dec_err)

    print(f"{model_name} Decryption mean: {dec_mean_low}ms-{dec_mean_high}ms, 95% CI")



# just use to make sure encrypted predictions are equivilant to plaintext predictions (they are)
def test_model_accuracy(model_name: str, enc: bool):
    if model_name[0] == 'z' and enc:
        zama_test_accuracy(model_name[5:]) # zama_log -> log
    elif model_name[0] == 'z' and not enc:
        zama_test_plain_accuracy(model_name[5:]) # zama_log -> log
    elif model_name[0] == 't' and enc:
        ts_test_accuracy(model_name[3:])
    elif model_name[0] == 't' and not enc:
        ts_test_plain_accuracy(model_name[3:])
    elif model_name[0] == 'p' and enc:
        pal_test_accuracy(model_name[4:])
    elif model_name[0] == 'p' and not enc:
        pal_test_plain_accuracy(model_name[4:])
    else:
        raise ValueError(f"model_name `{model_name}` should begin with `zama_` or `ts_`")


# potentially return accuracy values (instead of just printing?). Don't see why I need to do this? Just show once
# how accurate it is, and save it on paper...
def ts_test_accuracy(model_name):
    model = util.loadModelPickle(f"ts_plain_models/{model_name}")
    ctx_eval = client.setup_ts_params() # remember to change depending on what the model is (when we test log)

    is_log = True
    if model_name == "svm" or model_name == "pca_svm": # potentially make more robust. What if it's neither?
        is_log = False

    x, y = (red_X_rand, red_y_rand) if _is_svd_model(model_name) else (X_rand, y_rand)

    print("Encrypting test set...")
    enc_x = client.ts_encrypt_x(x, ctx_eval)
    print("Test set encrypted.")

    total = len(y_rand)
    correct_counter = 0

    print("Testing accuracy...")

    correct_count, accuracy = _ts_accuracy_test_loop(model, enc_x, y, is_log)

    print(f"\nTENSEAL {model_name} | CORRECT={correct_count} | HE ACCURACY={accuracy}%")

def pal_test_accuracy(model_name):
    pal_model = util.loadModelPickle(f"pal_plain_models/{model_name}")
    is_log = True
    if model_name == "svm" or model_name == "pca_svm": # potentially make more robust. What if it's neither?
        is_log = False

    x, y = (red_X_rand, red_y_rand) if _is_svd_model(model_name) else (X_rand, y_rand)

    public_key, private_key = client.pal_gen_keys()

    print("Encrypting test set...")
    s = time.perf_counter()
    enc_x = client.pal_enc_dataset(public_key, x)
    e = time.perf_counter()
    print(f"Test set of size {sample_size} encrypted in {e - s} seconds.")

    accuracy = _pal_accuracy_test_loop(pal_model, private_key, enc_x, y, is_log)
    print(f"\nPAILLIER {model_name} HE ACCURACY = {accuracy}%")


def _ts_accuracy_test_loop(model, enc_x, y, is_log) -> tuple:
    correct_counter = 0

    for i in range(sample_size):
        enc_prelim_y = model.enc_prelim_predict(enc_x[i])
        prelim_y = enc_prelim_y.decrypt()
        pred_y = None
        if is_log:
            pred_y = client.ts_client_finish_prediction_log(prelim_y)
            if client.ts_is_correct_prediction_log(y[i], pred_y):
                correct_counter += 1
        else:
            pred_y = client.ts_client_finish_prediction_svm(prelim_y)
            if client.ts_is_correct_prediction_svm(y[i], pred_y):
                correct_counter += 1

    return correct_counter, (correct_counter / sample_size)

def _pal_accuracy_test_loop(model, private_key, enc_x, y, is_log):
    correct_counter = 0

    for i in range(sample_size):
        enc_prelim_y = [model.enc_prelim_predict(enc_x[i])] # it's expecting y to be in a list (ts does this automatically. Pal doesn't)
        prelim_y = client.pal_decrypt_prelim(private_key, enc_prelim_y)
        pred_y = None
        if is_log:
            pred_y = client.ts_client_finish_prediction_log(prelim_y)
            if client.ts_is_correct_prediction_log(y[i], pred_y):
                correct_counter += 1
        else:
            pred_y = client.ts_client_finish_prediction_svm(prelim_y)
            if client.ts_is_correct_prediction_svm(y[i], pred_y):
                correct_counter += 1

    return correct_counter / sample_size

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
        pred_y = model.plaintext_predict(X_rand[i])
        #print(f"Actual y = {y_rand[i]} | predicted y = {pred_y}")
        if pred_y == y_rand[i]:
            correct_counter += 1

    print(f"\nTENSEAL {model_name} | CORRECT={correct_counter} | HE ACCURACY={correct_counter / total}%")

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

# i know this is horribly manual, but some reason desearilising keys freezes when i try to test here.
# so using printed results from sending 30 requests to server
def cpu_usage():
    usage = [97.9, 93.8, 95.7, 96.6, 93.2, 97.4, 96.4, 98.1, 98.1, 97.4,
             93.1, 87.4, 89.4, 93.6, 90.9, 98.7, 93.6, 98.4, 91.6, 97.6,
             90.7, 92.1, 96.8, 98.1, 92.4, 98.6, 97.8, 96.5, 89.3, 92.6]

    mean, sd, var, n = getStats(usage)
    printStats(mean, sd, var, n)
    err = calcErrorMargin(sd, n)

    mean_low, mean_high = getMeanRange(mean, err)

    print(f"CPU (%) mean: {mean_low}-{mean_high}, 95% CI")


# returns n timed inferences
def zama_benchmark_model(model_name) -> list:
    model = z.loadModel(model_name)
    model.load()

    inference_times = _zama_inference_test_loop(model, model_name, _is_svd_model(model_name))

    return inference_times

# the model doesn't actually matter here, but needed to get key specs
def zama_benchmark_enc_dec(model_name) -> tuple:
    cli, eval_keys = z.client(model_name)
    print("MODEL = ", model_name)
    model = z.loadModel(model_name)

    x = red_X_rand if _is_svd_model(model_name) else X_rand

    enc_times = []
    dec_times = []

    for i in range(sample_size):
        enc, dec = _zama_enc_dec_timers(model, x, cli, eval_keys, i)
        enc_times.append(enc)
        dec_times.append(dec)
        print(f"enc/dec {i} complete.")

    return enc_times[3:], dec_times[3:]

def _zama_enc_dec_timers(model, x, cli, eval_keys, i) -> tuple:
    s1 = time.perf_counter()
    enc_x = cli.quantize_encrypt_serialize(x[i:i+1]) # i:i+1 preserves (1,n) 2d-ness
    e1 = time.perf_counter()

    enc_time = e1 - s1

    # have the model output new result. Just decrypted straight after doesn't seem to be allowed? + prob more rigorous to do this way
    enc_result = model.run(enc_x, eval_keys)

    s2 = time.perf_counter()
    plain_x = cli.deserialize_decrypt_dequantize(enc_result) # not sure if decrypting without the enc_x changing is too fast? (like no server prediction to change result)
    e2 = time.perf_counter()

    dec_time = e2 - s2

    return enc_time, dec_time


# remove svm name after once we have log set up too (just to remind us we're solely testing svm)
def ts_benchmark_model(model_name):
    ts_model = util.loadModelPickle(f"ts_plain_models/{model_name}")

    x, ctx = (red_X_rand, client.setup_ts_params_pca()) if _is_svd_model(model_name) else (X_rand, client.setup_ts_params())
    enc_x = client.ts_encrypt_x(x, ctx)

    inference_times = _ts_or_pal_inference_test_loop(ts_model, enc_x)

    return inference_times

def ts_benchmark_enc_dec(model_name) -> tuple:
    x, ctx = (red_X_rand, client.setup_ts_params_pca()) if _is_svd_model(model_name) else (X_rand, client.setup_ts_params())
    model = util.loadModelPickle(f"ts_plain_models/{model_name}")

    enc_times = []
    dec_times = []

    for i in range(sample_size):
        enc, dec = _ts_enc_dec_timers(model, x, ctx, i)
        enc_times.append(enc)
        dec_times.append(dec)
        print(f"enc/dec {i} complete.")

    return enc_times[3:], dec_times[3:]

def _ts_enc_dec_timers(model, x, ctx_eval, i) -> tuple:
    s1 = time.perf_counter()
    enc_x = client.ts_encrypt_x_i(x[i], ctx_eval)
    e1 = time.perf_counter()
    enc_time = e1 - s1

    # have the model output new result. Just decrypted straight after doesn't seem to be allowed? + prob more rigorous to do this way
    enc_prelim_result = model.enc_prelim_predict(enc_x)

    s2 = time.perf_counter()
    plain_prelim_result = client.ts_decrypt(enc_prelim_result)
    e2 = time.perf_counter()
    dec_time = e2 - s2

    return enc_time, dec_time

def pal_benchmark_model(model_name):
    pal_model = util.loadModelPickle(f"pal_plain_models/{model_name}")
    x = red_X_rand if _is_svd_model(model_name) else X_rand

    public_key, private_key = client.pal_gen_keys()
    s = time.perf_counter()
    enc_x = client.pal_enc_dataset(public_key, x)
    e = time.perf_counter()
    print(f"Pal encrypting dataset of {sample_size} took {e - s} seconds.")

    inference_times = _ts_or_pal_inference_test_loop(pal_model, enc_x)

    return inference_times


def pal_benchmark_enc_dec(model_name) -> tuple:
    x = red_X_rand if _is_svd_model(model_name) else X_rand
    model = util.loadModelPickle(f"pal_plain_models/{model_name}")

    public_key, private_key = client.pal_gen_keys()

    enc_times = []
    dec_times = []

    for i in range(sample_size):
        enc, dec = _pal_enc_dec_timers(model, x, public_key, private_key, i)
        enc_times.append(enc)
        dec_times.append(dec)
        print(f"enc/dec {i} complete.")

    return enc_times[3:], dec_times[3:]

def _pal_enc_dec_timers(model, x, pub_key, priv_key, i):
    s1 = time.perf_counter()
    enc_x = client.pal_enc_x_i(pub_key, x[i])
    e1 = time.perf_counter()
    enc_time = e1 - s1

    # have the model output new result. Just decrypted straight after doesn't seem to be allowed? + prob more rigorous to do this way
    enc_prelim_result = model.enc_prelim_predict(enc_x)

    s2 = time.perf_counter()
    plain_prelim_result = client.pal_decrypt_prelim(priv_key, [enc_prelim_result])
    e2 = time.perf_counter()
    dec_time = e2 - s2

    return enc_time, dec_time



def zama_bench_plain_inference(model_name):
    inferenceTimes = []
    model = util.loadModelPickle(f"zama_plain_models/{model_name}")

    x = red_X_rand if _is_svd_model(model_name) else X_rand

    for i in range(len(X_rand)):
        inferenceTimes.append(zama_plain_inference(model, x[i:i+1]))
        print(f"Inference {i} complete.")

    return inferenceTimes[3:] # dropping first 3 elements

def ts_bench_plain_inference(model_name):
    inferenceTimes = []
    model = util.loadModelPickle(f"ts_plain_models/{model_name}")

    x = red_X_rand if _is_svd_model(model_name) else X_rand

    for i in range(len(X_rand)):
        inferenceTimes.append(ts_plain_inference(model, x[i]))
        print(f"Inference {i} complete.")

    return inferenceTimes[3:] # dropping first 3 elements

 

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


def zama_plain_inference(model, data) -> float:
    start = time.perf_counter()
    model.predict(data)
    end = time.perf_counter()
    return end - start

def ts_plain_inference(model, x_i):
    start = time.perf_counter()
    model.plaintext_predict(x_i)
    end = time.perf_counter()
    return end - start


# _name() functions here (like private) intended to be used within other functions.

def _is_svd_model(model_name) -> bool:
    return model_name[:3] == "svd"

def _zama_inference_test_loop(model, model_name, is_pca_model):
    cli, eval_keys = z.client(model_name)
    inference_times = []

    for i in range(sample_size):
        enc_x = None
        if is_pca_model:
            enc_x = cli.quantize_encrypt_serialize(red_X_rand[i:i+1]) # i:i+1 preserves (1,n) 2d-ness
        else:
            enc_x = cli.quantize_encrypt_serialize(X_rand[i:i+1])
        inference_times.append(zama_inference_time(model, enc_x, eval_keys))
        print(f"Inference {i} complete.")

    return inference_times[3:]

def _ts_or_pal_inference_test_loop(model, enc_x):
    inference_times = []
    for i in range(sample_size):
        start = time.perf_counter()
        model.enc_prelim_predict(enc_x[i]) # encrypted inference (don't need result)
        stop = time.perf_counter()
        inference_times.append(stop - start)
        print(f"Inference {i} complete.")

    return inference_times[3:] # dropping first 1 only for pal, since encryption is way too long..

if __name__ == "__main__":
    main()