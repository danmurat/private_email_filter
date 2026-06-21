import tenseal as ts
import src.util as util
from fastapi import FastAPI, File, Response, UploadFile

app = FastAPI()
# 'our server model'
ts_svd_svm = util.load_cloud_model("svd_svm")

"""
Run this file from root directory (backend/).
bash command: uvicorn src.main:app --reload
"""


@app.post("/spamfilter/ts")
async def encrypted_prediction(
    pub_ctx_file: UploadFile = File(...), enc_email_file: UploadFile = File(...)
):
    pub_ctx_bytes = await pub_ctx_file.read()  # should be python bin objs now
    enc_email_bytes = await enc_email_file.read()

    print(f"Received Encrypted email:\n\n {enc_email_bytes[:200]}.....\n")

    pub_ctx = ts.context_from(pub_ctx_bytes)
    enc_email = ts.ckks_vector_from(pub_ctx, enc_email_bytes)

    enc_prelim_result = ts_svd_svm.enc_prelim_predict(enc_email)
    enc_presult_bytes = enc_prelim_result.serialize()

    print(f"Encrypted result:\n\n {enc_presult_bytes[:100]}.....\n")

    return Response(content=enc_presult_bytes, media_type="application/octet-stream")
