import pickle
import numpy as np
import random
import pandas as pd
from typing import Any

reduced_data_path: str = "data/reduced_model_data"
test_data_path: str = "data/test.jsonl"

def load_red_test_data() -> Any:
    reduced_model_data = _load_model_pickle(reduced_data_path)

    return reduced_model_data.get_red_X_test()

def randomise(n: int, X) -> tuple:
    # these might have to be changed to an np.array
    randomised_X = np.zeros((n, len(X[0])))
    rand = None
    for i in range(n):
        rand = random.randint(0, 1999)  # 2000 is out of bounds..
        randomised_X[i] = X[rand]

    return randomised_X, rand

def print_selected_test_email(index: int) -> None:
    test_emails = pd.read_json(test_data_path, lines=True)
    # print(test_emails)
    selected_email = test_emails.loc[index]["text"]

    print(f"Email:\n\n {selected_email}\n")

def _load_model_pickle(path: str) -> Any:
    with open(f"{path}.pkl", "rb") as file:
        return pickle.load(file)

