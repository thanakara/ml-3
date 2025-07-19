from dataclasses import dataclass
from omegaconf import ListConfig


@dataclass
class Dense:
    in_units: int
    out_units: int
    kernel_regularizer: str | None=None
    bias_regularizer: str | None=None
    activity_regularizer: str | None=None
    kernel_constraint: str | None=None
    bias_constraint: str | None=None
    activation: str="relu"
    use_bias: bool=True
    kernel_initializer: str="glorot_uniform"
    bias_initializer: str="zeros"

@dataclass
class Conv2D:
    input_shape: tuple | list | None
    filters: int
    kernel_size: int
    strides: int=1
    padding: str="valid"
    data_format: str=None
    dilation_rate: int=1
    groups: int=1
    activation: str=None
    use_bias: bool=True
    kernel_initializer: str="glorot_uniform"
    bias_initializer: str="zeros"
    kernel_regularizer: str | None=None
    bias_regularizer: str | None=None
    activity_regularizer: str | None=None
    kernel_constraint: str | None=None
    bias_constraint: str | None=None


def get_out_units(shape: list[int]) -> int:
    num_input_fmaps = 1

    for s in shape:
        num_input_fmaps *= s
        
    return num_input_fmaps

def calc_channels(*args):
    (args,) = args
    height, *_ = args.get("input_shape")
    filters = args.get("filters")
    padding = args.get("padding")
    kernel = args.get("kernel_size")
    strides = args.get("strides")

    if padding == "valid":
        pad = 0
    elif padding == "same":
        pad = (kernel - 1) / 2
    else:
        raise TypeError
    
    channel = int(
        ((height + 2*pad - kernel) / strides) + 1
    )

    return ListConfig([channel, channel, filters])