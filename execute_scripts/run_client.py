import requests
import util
import client
import tenseal as ts

base_url = "http://127.0.0.1:8000/"
X_test, red_X_test, y_test = util.load_test_data()
x_i, y = util.randomise(1, X_test, y_test)

def main():
    try:
        data, ctx = ts_data_to_send()
        response = requests.post(
            base_url + "spamfilter/ts",
            files=data
        )
        response.raise_for_status() # incase error (not sure why this is needed. thought error would just throw)

        if response.status_code == 200:
            enc_prelim_result_bytes = response.content
            enc_prelim_result = ts.ckks_vector_from(ctx, enc_prelim_result_bytes)
            prelim_result = enc_prelim_result.decrypt()

            result = client.ts_client_finish_prediction_svm(prelim_result)
            if result == 0:
                print("HAM")
            else:
                print("SPAM")



    except requests.exceptions.HTTPError as err:
        print(f"HTTP error: {err}")
    except Exception as err:
        print(f"error: {err}")


def ts_data_to_send() -> tuple:
    ctx = client.setup_ts_params()
    enc_x_i = client.ts_encrypt_x_i(x_i[0], ctx)

    enc_email_bytes = enc_x_i.serialize()

    pub_ctx_bytes = ctx.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_relin_keys=True,
        save_galois_keys=True
    )

    tuple_data = {
        "pub_ctx": ("pub_ctx.bin", pub_ctx_bytes, "application/octet-stream"),
        "enc_email": ("enc_email.bin", enc_email_bytes, "application/octet-stream")
    }

    return tuple_data, ctx

if __name__ == "__main__":
    main()