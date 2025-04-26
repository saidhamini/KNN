from pydantic import BaseModel, conint
from typing import Optional, Literal

class HyperParams(BaseModel):
    n_neighbors: conint(gt=0) = 5  # must be > 0
    weights: Literal['uniform', 'distance'] = 'uniform'
    algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = 'auto'
    leaf_size: conint(gt=0) = 30   # must be > 0
    p: conint(gt=0) = 2            # typically 1 or 2
    metric: Literal['minkowski', 'euclidean', 'manhattan'] = 'minkowski'
