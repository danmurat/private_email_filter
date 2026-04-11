import numpy as np
import tenseal as ts
import phe as paillier
"""
Hypotheically, code here represents what the client alone can do. For tenseal models, the server
would give a preliminary prediction (encrypted), and sent back to the client. To keep things efficient, that is
all the server would need to do. Depending on the model (svm/log), the client would finish this with whatever
else is needed mathematically (sigmoid for logistic, sign for svm).

Even though all the code is in the same repo, on the same machine; i'm seperating to maintain the concept and "simulate"
what would actually occur in a client-server model.
"""

# assumes y_prelim has been decrypted by client
def ts_client_finish_prediction_log(y_prelim: list):
    return _sigmoid(y_prelim[0])

# gives number between 0, 1 (with 0.5 being the boundary)
def _sigmoid(x):
    return 1 / (1 + np.exp(-x)) # without brackets, there's a sneaky bidmas error!!!

# if difference > 0.5, we know that the prediction is predicting the opposite classification
def ts_is_correct_prediction_log(y, y_pred):
    #print(f"Actual label = {y} | Predicted label = {y_pred}")
    return np.abs(y - y_pred) < 0.5

# all client has to do is check if the value is positive or negative. Here we just make them equal
# to their correct labels for testing
def ts_client_finish_prediction_svm(y_prelim):
    # y_prelim is a list wth 1 value (depending on how we're encrypting!) Right now we have encrypted value (not vector)
    y = 1 if y_prelim[0] >= 0 else 0

    return y

def ts_is_correct_prediction_svm(y, y_pred):
    #print(f"Actual label = {y} | Predicted label = {y_pred}")
    return y_pred == y

"""
IMPORTANT

Making sure these params are correct allows the actual encrypted operations to work. Reason why we were
wildly off was because our input wasn't fitting in the cyphertext, so was missing a lot of info!

The numbers given below also determine how efficient and secure the scheme runs. How does it work? I don't know.
I guess I'll play around with the numbers at first, until I can find some known optimisations to place.

PCA'd data may be interesting here? for it to run VERY fast?

NOTE: there may be better param options depending on svm/log too. For now keep as the same
"""
def setup_ts_params():
    # from tenseal docs:
    # parameters
    poly_mod_degree = 2 ** 14 # 2^12 = 4096 basic (must be pow 2) -- lower = more efficient
    coeff_mod_bit_sizes = [40, 40, 40, 40]
    # create TenSEALContext
    ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    # scale of ciphertext to use
    ctx_eval.global_scale = 2 ** 40
    # this key is needed for doing dot-product operations
    ctx_eval.generate_galois_keys()

    return ctx_eval

# testing. We should be able to have lower values for this?? We go from dimensions=5000+ to 200
# this is giving us ~4ms inference. We can't seem to go lower than 2^12 though?
def setup_ts_params_pca():
    poly_mod_degree = 2 ** 12 # 2^12 = 4096 basic (must be pow 2) -- lower = more efficient
    coeff_mod_bit_sizes = [40, 20, 40]
    ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    ctx_eval.global_scale = 2 ** 10
    ctx_eval.generate_galois_keys()

    return ctx_eval

# x represents a whole matrix. x_i for single row of the matrix
def ts_encrypt_x(x, ctx_eval):
    enc_x = [ts.ckks_vector(ctx_eval, i.tolist()) for i in x]
    return enc_x

# assumes x_i has been flattened to a vector (1 dimensional)
def ts_encrypt_x_i(x_i, ctx_eval):
    enc_x_i = ts.ckks_vector(ctx_eval, x_i)
    return enc_x_i


# -- Paillier stuff --

# key length < 4096 for now, since that's too slow
def pal_gen_keys() -> tuple:
    return paillier.generate_paillier_keypair(n_length=1024)

def pal_enc_x_i(public_key, x_i):
    return [public_key.encrypt(n) for n in x_i]

def pal_decrypt_prelim(private_key, enc_prelim_y):
    return [private_key.decrypt(n) for n in enc_prelim_y]