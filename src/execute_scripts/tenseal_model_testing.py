import torch
import spam_imp.src.util as util
import tenseal as ts
import spam_imp.src.client as client
from spam_imp.src.TenSealModels import TenSealModels
from spam_imp.src.data_functionality.PreProcess import PreProcess
from spam_imp.src.ts_compat_models import LinearSVM

"""
Any random testing needed is done in here.
"""

model_data = util.loadModelPickle(util.model_data_path())
reduced_model_data = util.loadModelPickle(util.reduced_model_data_path())

X_train, y_train, X_test, y_test = model_data.get_all_data()
reduced_X_train, reduced_X_test = reduced_model_data.get_all_data()

# print(X_train)
# print(y_train)

# convert data to torch tensors
t_X_train, t_y_train, t_X_test, t_y_test, t_red_X_train, t_red_X_test = util.convertToTorchTensors(X_train, y_train, X_test, y_test, reduced_X_train, reduced_X_test)

dimensions = X_train.shape[1]

t = TenSealModels()

def main():
    #testLR()
    #testSVM()

    # apr 9th: testing tenseal encryption inference

    # encryption operations working now (encryption params wern't big enough)
    for i in range(10):
        test_enc_svm()

    # counter = 0
    # dupe_emails = []
    # for i in range(10):
    #     y_pred, email = test_enc_svm()
    #     dupe_emails.append(email)
    #     if len(dupe_emails) >= 2:
    #         if dupe_emails[0].all() != dupe_emails[1].all():
    #             raise ValueError("YOUR PREPROCESSING IS FUCKED")
    #         else:
    #             del dupe_emails[0]
    #     if y_pred == 0: counter += 1
    #print(f"Accuracy on same email: {counter/10}%")


# PROBLEM: plaintext prediction always gets the prediction right on the same email
# ENCRYPTING DOESN'T!!! It's correct like 70% of the time? BELOW 50% ???!
# why are we getting different results?!?
#
# first guess is that w/b should probably be encrypted too?
def test_enc_svm():
    print("loading ts svm...")
    svm = util.loadModelPickle("ts_plain_models/svm")
    print("model loaded.")

    p = PreProcess() # just for pre-processing single email

    ctx_eval = client.setup_ts_params_svm()
    indexed100_dict = util.getIndexedDict()
    # spam_email_test = util.load_single_spam_email_df_test(27070)
    # print(spam_email_test)
    spam_email = util.load_single_spam_email_df(902)
    #print(spam_email)

    spam_email_label = spam_email["label"].iloc[0]

    # print(f"email label = {spam_email_label}")
    # print("\n", spam_email["text"].iloc[0])

    spam_email_vector = p.preprocessSingleEmail(spam_email, indexed100_dict).flatten()
    #print("\n", spam_email_vector[:30])

    print("encrypting email...")
    # enc_X_test = [ts.ckks_vector(ctx_eval, x.tolist()) for x in t_X_test] # whole test set
    #print(f"spam vector = {spam_email_vector.flatten()}")
    enc_X_i = client.ts_encrypt_x_i(spam_email_vector, ctx_eval)
    print("test email encrypted.\n")
    print(f"Encrypted email:\n\n{enc_X_i.data}\ntruncated...\n")

    enc_prelim_y = svm.enc_prelim_predict(enc_X_i) # why are we getting different prelim values each time?
    #prelim_y = svm.enc_prelim_predict(spam_email_vector) # unencrypted prelim calc gives us same value each time (1.9 something)
    prelim_y = enc_prelim_y.decrypt() # client decrypts (1.9 something too now, though it's an approximation)
    print(f"prelim_y = {prelim_y}")
    y_pred = client.ts_client_finish_prediction_svm(prelim_y)

    # testing plain model. For some reason re-running sometimes gives an incorrect prediction? Even though
    # we load the same model, with the same data, with the same preprocessing steps?
    #y_pred = svm.predict(spam_email_vector)

    print(f"label = {spam_email_label} | predicted label = {y_pred}")


"""
IMPORTANT

Making sure these params are correct allows the actual encrypted operations to work. Reason why we were
wildly off was because our input wasn't fitting in the cyphertext, so was missing a lot of info!

The numbers given below also determine how efficient and secure the scheme runs. How does it work? I don't know.
I guess I'll play around with the numbers at first, until I can find some known optimisations to place.

PCA'd data may be interesting here? for it to run VERY fast?
"""
def _setup_ts_params_svm():
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



def testLR():
    lr = t.trainLog(t_X_train, t_y_train, 3000)

    t.logAccuracy(lr, t_X_test, t_y_test)

def testPcaLR():
    pca_lr = t.trainLog(t_red_X_train, t_y_train, 3000)

    t.logAccuracy(pca_lr, t_X_test, t_y_test)




# C at 200 - 800 pretty much all get 97% accuracy
def testSVM():
    print("training svm with c=200.0")
    svm = t.trainSVM(t_X_train, t_y_train)

    t.svmAccuracy(svm, t_X_test, t_y_test)

def testTenSeal():
    t.test()

if __name__ == "__main__":
    main()