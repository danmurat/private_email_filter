from unittest.mock import MagicMock, patch
from src.data_functionality.PreProcess import PreProcess

import pandas as pd
import pytest

pp = PreProcess()

first_error_emails = ["email 1"] + ["email 2"]
second_error_emails = ["email 1"] + ["email 2"] + ["email 3"] + ["email 4"]


# ensure that we process 1 email
@pytest.mark.parametrize(
    "bad_input",
    [
        pd.Series(dtype=str),
        pd.Series(first_error_emails),
        pd.Series(second_error_emails),
    ],
)
def test_preprocess_single_email_error(bad_input):
    with pytest.raises(
        ValueError,
        match=f"Your email: pd.Series must have 1 email only! You have {bad_input.size}",
    ):
        pp.preprocess_single_email_tfidf(bad_input)


@patch("src.util.load_model_pickle")
def test_preprocess_single_email_success(mock_load_pickle):
    fake_tfidf = MagicMock()
    fake_tfidf.transform.return_value = "fake_sparse_matrix"
    mock_load_pickle.return_value = fake_tfidf

    good_input = pd.Series(["some email"])
    result = pp.preprocess_single_email_tfidf(good_input)

    assert result == "fake_sparse_matrix"
