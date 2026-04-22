from fastapi import FastAPI, UploadFile, File, Response
import util
import tenseal as ts

app = FastAPI()
# 'our server model'
ts_svm = util.loadModelPickle("ts_plain_models/svm")

@app.post("/spamfilter/ts")
async def encrypted_prediction(
        pub_ctx: UploadFile = File(...),
        enc_email: UploadFile = File(...)
):
    pub_ctx_bytes = await pub_ctx.read()     # should be python bin objs now
    enc_email_bytes = await enc_email.read()

    pub_ctx = ts.context_from(pub_ctx_bytes)
    enc_email = ts.ckks_vector_from(pub_ctx, enc_email_bytes)

    enc_prelim_result = ts_svm.enc_prelim_predict(enc_email)
    enc_presult_bytes = enc_prelim_result.serialize()


    return Response(content=enc_presult_bytes, media_type="application/octet-stream")