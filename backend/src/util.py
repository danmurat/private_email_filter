from typing import Any
import cloudpickle

def load_cloud_model(name: str) -> Any:
    with open(f"models/{name}.pkl", "rb") as file:
        return cloudpickle.load(file)
