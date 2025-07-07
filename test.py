import numpy as np
import tensorflow as tf
import torch
import os
from ml_3.model.initializers.initializer import lecun_normal_
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


np.random.seed(42)
torch.manual_seed(42)
# tf.keras.utils.set_random_seed(42)
M = np.random.rand(2, 2).astype(np.float32)
torch_tensor = torch.tensor(M, dtype=torch.float32)

def np_lecun(shape):
    fan_in = np.prod(shape[1:])
    std = np.sqrt(1.0 / fan_in, dtype=np.float32)
    return np.random.normal(0.0, std, shape)

print(np_lecun(M.shape))
print()
print(lecun_normal_(torch_tensor).numpy())
print()
print(tf.keras.initializers.lecun_normal(42)(M.shape).numpy())