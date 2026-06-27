import sys

import requests
import client_code
import util
import tenseal as ts

#base_url = "http://0.0.0.0:80/" # nginx listening on port 80
aws_url = "http://35.178.170.58/"
red_X_test = util.load_red_test_data()
# TESTING ON SVD DATA
x_i, index = util.randomise(1, red_X_test)


def main() -> None:
    util.print_selected_test_email(index)

    try:
        choice = input("\nEncrypt email? (y): \n")
        if choice == "n":
            sys.exit()

        data, ctx = _ts_data_to_send()

        choice = input("\nSend for classification? (y): \n")
        if choice == "n":
            sys.exit()

        response = requests.post(aws_url + "spamfilter/ts", files=data)
        response.raise_for_status()  # incase error (not sure why this is needed. thought error would just throw)

        if response.status_code == 200:
            enc_prelim_result_bytes = response.content
            print(f"Encrypted result:\n\n {enc_prelim_result_bytes[:100]}\n")
            enc_prelim_result = ts.ckks_vector_from(ctx, enc_prelim_result_bytes)

            choice = input("\nDecrypt result? (y): \n")
            if choice == "n":
                sys.exit()

            prelim_result = enc_prelim_result.decrypt()
            result = client_code.ts_client_finish_prediction_svm(prelim_result)
            if result == 0:
                print("HAM")
            else:
                print("SPAM")

    except requests.exceptions.HTTPError as err:
        print(f"HTTP error: {err}")
    except Exception as err:
        print(f"error: {err}")


def _ts_data_to_send() -> tuple:
    ctx = client_code.setup_lean_ts_params() # USE LEAN! SERVER USES THE OPTIMIZED MODEL AND THIS DRASTICALLY REDUCES pub_ctx file size!
    enc_x_i = client_code.ts_encrypt_x_i(x_i[0], ctx)

    enc_email_bytes = enc_x_i.serialize()

    pub_ctx_bytes = ctx.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_relin_keys=True,
        save_galois_keys=True,
    )

    print(f"Encrypted email:\n\n {enc_email_bytes[:200]}.....\n")

    # Size checks in megabytes
    enc_email_mb = len(enc_email_bytes) / (1024 * 1024)
    pub_ctx_mb = len(pub_ctx_bytes) / (1024 * 1024)
    print(f"--- Data Sizes ---")
    print(f"Encrypted Email: {enc_email_mb:.3f} MB")
    print(f"Public Context (Keys): {pub_ctx_mb:.3f} MB")
    print(f"Total Upload Size: {enc_email_mb + pub_ctx_mb:.3f} MB\n")

    tuple_data = {
        "pub_ctx_file": ("pub_ctx.bin", pub_ctx_bytes, "application/octet-stream"),
        "enc_email_file": (
            "enc_email.bin",
            enc_email_bytes,
            "application/octet-stream",
        ),
    }

    return tuple_data, ctx


# for demo:
# need to print the contents! Both client and server to showcase what's happening.

if __name__ == "__main__":
    main()
