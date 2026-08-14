### Setup

This backend is simply for serving an inference result back to the client. You'll see
`models\` and `data` here which may be unconventional. But all we need is our slim SVM tenseal
model + test data to test a random encrypted prediction.

The model is 4kb. Trivial to say the least, so might as well include
it in the backend dir and docker container. This will greatly simplify server inference (I
won't need to access far reaching directories, or make sure I send a trained model over
to Amazon's storage whatever thingies so the server can access it. All that's needed will
simply be here, already, in a nice, lean package).

### TODO

Since we're seperating the backend from the ml, we'll need to handle preprocessing
here too (just copy paste what's needed for a single email. Preprocessing only needed
if a user sends their own pasted email in).

## Browser raw-SEAL endpoint

`POST /spamfilter/seal` accepts raw Microsoft SEAL serialization from the
browser (`parameters_file`, `galois_keys_file`, and `enc_email_file`, plus
protocol metadata) and returns a raw ciphertext. The server never receives the
client secret key. Set `CORS_ORIGINS` to the deployed HTTPS frontend origin(s)
instead of the development default `*`.

The lean CKKS profile used by the demo is intentionally small and uses
`security_level=-1`; it must not be presented as production-strength
cryptography. Deploy the API behind HTTPS and review the parameters before any
real data is processed.
