import sys
import time

import numpy as np
from HandleModel import HandleModel
from PreProcess import PreProcess

import spam_imp.src.util as util

p = PreProcess()
# p.preprocess() # including these just for testing against whole dataset
h = HandleModel(p)
# h.dataSetup()

# index = 2 is an pre-ejaculation spam email. This might be good to demo lol
# we should probably do 2 examples. One spam, one not
# def testSpam():
#     p.preprocessSingleEmail(2, True)

# # these indexes arn't split 0..2000! If i access [2] it's an error because it's spam, not ham!
# def testHam():
#     p.preprocessSingleEmail(1998, False)

# def load():
#     return h.loadModel("svm")


def handle_loading(message: str) -> None:
    dots = [".", "..", "..."]
    for i in range(12):
        print(f"\r{message}{dots[i % 3]}", end="", flush=True)
        time.sleep(0.25)


def print_result(result: np.intp) -> None:
    if result == 0:
        print("\n\nPrediction: NOT SPAM!")
    else:
        print("\n\nPrediction: SPAM!")


# we need to grab 1 spam message, show it, preproccess it (ALONE), then do the encrypting

"""
Encrypted predictions now working from saved model weights!

Below has to be cleaned up dramatically, and flow through the demo. So we need some 
pauses and questions ("send to server?" for example). Should completely be ran in steps
(slow but not too slow, we only have 10mins to speak).
"""

# TODO: add an "inference: {result}seconds" when classifying
if __name__ == "__main__":
    # handleLoading("Recieving email")

    # ham_email = p.loadSingleHamEmail(7311)
    # ham_email_label = ham_email["label"].iloc[0] # of course = 0

    # print("Email: \n", ham_email["text"].iloc[0])

    # pass these into preprocess (single), so it returns vectorised set.

    # these are done "under the hood"
    indexed100 = util.get_indexed_dict()  # working!
    # ham_email_vector = p.preprocessSingleEmail(ham_email, indexed100) # final vectorised format

    # print("ham vector shape = ", ham_email_vector.shape)

    cli, eval_keys = h.client("svm")

    # choice = input("\nWould you like to encrypt the email? (y/n): \n")
    # if choice == "n":
    #     sys.exit()

    # handleLoading("Encrypting email")
    # enc_ham_data = cli.quantize_encrypt_serialize(ham_email_vector)
    # print(f"\n\nEncrypted data: \n\n {enc_ham_data[:350]}\n...truncated...")

    # # we pass this into the model, and the model will give us a result "spam"/"ham" and
    # # show this

    # choice = input("\nSend to server for classification? (y/n): \n")
    # if choice == "n":
    #     sys.exit()

    # handleLoading("Sending encrypted data")
    server_model = h.loadModel("svm")
    server_model.load()

    # # above and below loading conjoin..
    # print("\n")

    # handleLoading("classifying")
    # enc_result = server_model.run(enc_ham_data, eval_keys)
    # print("\nResult recieved!\n")
    # print(f"\nEncrypted result: \n\n {enc_result[:350]}\n...truncated...")

    # choice = input("\nDecrypt result? (y/n): \n")
    # if choice == "n":
    #     sys.exit()

    # handleLoading("decrypting result")
    # result_probabilities = cli.deserialize_decrypt_dequantize(enc_result)
    # result = np.argmax(result_probabilities)
    # printResult(result)

    # choice = input("\n\nContinue? (y/n): \n")

    # if choice == "n":
    #     sys.exit()

    # handles same as above but for spam

    handle_loading("Recieving email")

    spam_email = util.load_single_spam_email_df(27070)
    spam_email_label = spam_email["label"].iloc[0]
    print("\n", spam_email["text"].iloc[0])

    spam_email_vector = p.preprocessSingleEmail(spam_email, indexed100)

    choice = input("\nWould you like to encrypt the email? (y/n): \n")
    if choice == "n":
        sys.exit()

    handle_loading("Encrypting email")
    enc_spam_data = cli.quantize_encrypt_serialize(spam_email_vector)
    print(f"\n\nEncrypted data: \n\n {enc_spam_data[:350]}\n...truncated...")

    choice = input("\nSend to server for classification? (y/n): \n")
    if choice == "n":
        sys.exit()

    handle_loading("Sending encrypted data")
    enc_spam_result = server_model.run(enc_spam_data, eval_keys)
    print("\nresult recieved!\n")
    print(f"\nEncrypted result: \n\n {enc_spam_result[:350]}\n...truncated...")

    choice = input("\nDecrypt result? (y/n): \n")
    if choice == "n":
        sys.exit()

    handle_loading("Decrypting")
    spam_result_probs = cli.deserialize_decrypt_dequantize(enc_spam_result)
    spam_result = np.argmax(spam_result_probs)
    print_result(spam_result)
