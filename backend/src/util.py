from typing import Any
import pickle
import os
import sys

# because pickle keeps path from where file was CALLED (in code) rather than SAVED, we have
# to point back to the path where we called save_model_pickle() when training models.
ml_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../", "ml")
)

if ml_dir not in sys.path:
    sys.path.append(ml_dir)

def load_model_pickle(name: str) -> Any:
    with open(f"ml/models/pickle_objects/{name}.pkl", "rb") as file:
        return pickle.load(file)