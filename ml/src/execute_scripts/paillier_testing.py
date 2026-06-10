import time

import phe as paillier
import src.client as client
import src.util as util
from src.data_functionality.PreProcess import PreProcess

model_data = util.load_model_pickle(util.model_data_path())
reduced_model_data = util.load_model_pickle(util.reduced_model_data_path())

X_train, y_train, X_test, y_test = model_data.get_all_data()
reduced_X_train, reduced_X_test = reduced_model_data.get_all_data()


def main() -> None:
    # test_encryption_time()
    test_model("svm")


"""
Encryption grows linear to input size. ~0.38 seconds per item.

For vector of size 5177, this will take ~32 minutes..
pca'd vector (size 200), would take ~1 minute...

Already, paillier is pretty much infeasable? Unless there's a way to dramatically speed this up?

n_length at 2048 = ~0.13 sec
            1024 = ~0.01 sec
            
Not sure if these make the scheme insecure in any way however.
Also, we can apparently just install gmpy2, and the library will automatically use (to run c code for encryption stuff)

Yep.. gmpy2 game changer. Default key size ~0.017 seconds per item (1min 30 seconds for og data, 3.4 seconds for pca)
"""


def test_encryption_time() -> None:
    vec = [1]

    start1 = time.perf_counter()
    public_key, private_key = paillier.generate_paillier_keypair(
        n_length=2048
    )  # apparantly defaults to 4096
    end1 = time.perf_counter()

    print(f"key generation took {end1 - start1} seconds.")

    start2 = time.perf_counter()
    enc_vec = [public_key.encrypt(i) for i in vec]
    # enc_vec = public_key.encrypt(vec) # does a whole list input work? no
    end2 = time.perf_counter()

    print(f"encrypting vector of length {len(vec)} took {end2 - start2} seconds")


def test_model(model_name: str) -> None:
    pal_model = util.load_model_pickle(f"pal_plain_models/{model_name}")
    print(len(pal_model.w))

    p = PreProcess()

    indexed100_dict = util.get_indexed_dict()
    spam_email = util.load_single_spam_email_df(904)

    spam_email_label = spam_email["label"].iloc[0]

    spam_email_vector = p.preprocess_single_email(spam_email, indexed100_dict).flatten()

    print("Generating paillier keys...")
    public_key, private_key = client.pal_gen_keys()
    print("Keys generated.\n")

    print("Encrypting email...")
    start = time.perf_counter()
    enc_x_i = client.pal_enc_x_i(public_key, spam_email_vector)
    end = time.perf_counter()
    print(f"Email encrypted in {end - start} seconds.")

    print("Server evaluating encrypted email...")
    enc_prelim_y = [pal_model.enc_prelim_predict(enc_x_i)]  # add back to list
    print("Encrypted result returned.")

    prelim_y = client.pal_decrypt_prelim(private_key, enc_prelim_y)
    pred_y = client.ts_client_finish_prediction_log(prelim_y)

    print(f"Actual label = {spam_email_label} | predicted = {pred_y}")


if __name__ == "__main__":
    main()
