binary_data = ts_data_to_send()
response = requests.post(
        base_url + "spamfilter/",
        files=binary_data
        )

if response.status_code == 200:
    enc_result_bytes = response.content
    enc_result = enc_result_bytes.deserialize(private_key)
    result = enc_result.decrypt()
