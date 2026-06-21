from src.util import load_cloud_model

def test_load_cloud_model():
    # Verify that the SVD SVM model can be successfully loaded 
    # and has the necessary predicting capabilities.
    model = load_cloud_model("svd_svm")
    assert model is not None
    assert hasattr(model, "enc_prelim_predict")
