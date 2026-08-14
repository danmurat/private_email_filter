import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import tenseal as ts
from tenseal import sealapi
import src.util as util
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# 'our server model'
ts_svd_svm = util.load_cloud_model("svd_svm")

cors_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["*"],
)

SEAL_PROTOCOL_VERSION = "1"
SEAL_VECTOR_LENGTH = 150
SEAL_POLY_MODULUS_DEGREE = 4096
SEAL_COEFF_MODULUS_BITS = (40, 29, 40)
SEAL_SCALE = 2**29
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

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


async def _read_seal_upload(upload: UploadFile, name: str) -> bytes:
    """Read one raw-SEAL upload, rejecting empty or oversized files."""
    contents = await upload.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail=f"{name} is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"{name} exceeds the upload limit")
    return contents


def _validate_seal_parameters(parameters: sealapi.EncryptionParameters) -> None:
    if parameters.scheme() != sealapi.SCHEME_TYPE.CKKS:
        raise HTTPException(status_code=400, detail="Only CKKS parameters are accepted")
    if parameters.poly_modulus_degree() != SEAL_POLY_MODULUS_DEGREE:
        raise HTTPException(status_code=400, detail="Invalid polynomial modulus degree")
    coeff_bits = tuple(modulus.bit_count() for modulus in parameters.coeff_modulus())
    if coeff_bits != SEAL_COEFF_MODULUS_BITS:
        raise HTTPException(status_code=400, detail="Invalid coefficient modulus")


@app.post("/spamfilter/seal")
async def raw_seal_prediction(
    parameters_file: UploadFile = File(...),
    galois_keys_file: UploadFile = File(...),
    enc_email_file: UploadFile = File(...),
    protocol_version: str = Form(...),
    vector_length: int = Form(...),
):
    """Evaluate a node-seal ciphertext using raw Microsoft SEAL serialization.

    The three binary fields are deliberately raw SEAL objects rather than
    TenSEAL protobuf wrappers.  The client retains the secret key and only
    sends evaluator material to this endpoint.
    """
    if protocol_version != SEAL_PROTOCOL_VERSION:
        raise HTTPException(status_code=400, detail="Unsupported SEAL protocol version")
    if vector_length != SEAL_VECTOR_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid encrypted vector length")

    parameters_bytes, galois_bytes, ciphertext_bytes = await asyncio.gather(
        _read_seal_upload(parameters_file, "parameters_file"),
        _read_seal_upload(galois_keys_file, "galois_keys_file"),
        _read_seal_upload(enc_email_file, "enc_email_file"),
    )
    if len(parameters_bytes) + len(galois_bytes) + len(ciphertext_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Combined upload exceeds the limit")

    with TemporaryDirectory(prefix="privmail-seal-") as temp_dir:
        temp_path = Path(temp_dir)
        parameters_path = temp_path / "parameters.bin"
        galois_path = temp_path / "galois-keys.bin"
        ciphertext_path = temp_path / "ciphertext.bin"
        result_path = temp_path / "result.bin"
        parameters_path.write_bytes(parameters_bytes)
        galois_path.write_bytes(galois_bytes)
        ciphertext_path.write_bytes(ciphertext_bytes)

        try:
            parameters = sealapi.EncryptionParameters(sealapi.SCHEME_TYPE.CKKS)
            parameters.load(str(parameters_path))
            _validate_seal_parameters(parameters)
            context = sealapi.SEALContext(
                parameters, True, sealapi.SEC_LEVEL_TYPE.NONE
            )
            if not context.parameters_set():
                raise HTTPException(status_code=400, detail="Invalid SEAL parameters")

            galois_keys = sealapi.GaloisKeys()
            galois_keys.load(context, str(galois_path))
            if list(galois_keys.parms_id()) != list(context.key_parms_id()):
                raise HTTPException(status_code=400, detail="Galois keys are for the wrong context")
            ciphertext = sealapi.Ciphertext()
            ciphertext.load(context, str(ciphertext_path))

            if ciphertext.poly_modulus_degree() != SEAL_POLY_MODULUS_DEGREE:
                raise HTTPException(status_code=400, detail="Invalid ciphertext modulus degree")
            # SEAL stores a freshly encrypted CKKS ciphertext at the first
            # usable chain level, which has one fewer modulus than the
            # parameter list in this profile.
            if ciphertext.coeff_modulus_size() not in {
                len(SEAL_COEFF_MODULUS_BITS) - 1,
                len(SEAL_COEFF_MODULUS_BITS),
            }:
                raise HTTPException(status_code=400, detail="Invalid ciphertext modulus")
            if abs(ciphertext.scale - SEAL_SCALE) > 1e-6:
                raise HTTPException(status_code=400, detail="Invalid ciphertext scale")
            if list(ciphertext.parms_id()) != list(context.first_parms_id()):
                raise HTTPException(status_code=400, detail="Ciphertext is at the wrong level")

            encoder = sealapi.CKKSEncoder(context)
            evaluator = sealapi.Evaluator(context)
            weights = [float(value) for value in ts_svd_svm.w]
            bias = float(ts_svd_svm.b[0])
            if len(weights) != SEAL_VECTOR_LENGTH:
                raise RuntimeError("The loaded SVM model has an invalid dimension")

            weight_plain = sealapi.Plaintext()
            encoder.encode(weights, SEAL_SCALE, weight_plain)
            weighted = sealapi.Ciphertext()
            evaluator.multiply_plain(ciphertext, weight_plain, weighted)

            """The client packs the 150 SVD values into the first 150 CKKS
            slots.  
            The power-of-two rotate-and-add algo reduction sums those slots.
            
            e.g [1,2,3,4] this sums to 10
            rotating shifts everyting left by the rotator amount
            1: [1+2, 2+3, 3+4, 4+0]
            2: [1+2+3+4, ....]
            See how the first element contains the correct sum? This is how the algorithm works, combining
            the partial additions in a tree-like manner, till you get to the final half (1+2 and the 3+4) to
            combine the final answer.
            """
            reduced = weighted
            for rotation in (1, 2, 4, 8, 16, 32, 64, 128):
                rotated = sealapi.Ciphertext()
                evaluator.rotate_vector(reduced, rotation, galois_keys, rotated)
                evaluator.add_inplace(reduced, rotated)

            bias_plain = sealapi.Plaintext()
            encoder.encode(bias, reduced.scale, bias_plain)
            evaluator.sub_plain(reduced, bias_plain, reduced)
            reduced.save(str(result_path))
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"Invalid SEAL payload: {error}") from error

        return Response(content=result_path.read_bytes(), media_type="application/octet-stream")
