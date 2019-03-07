from math import floor


def output_size_conv2d_layer(height, width, layer):
    kernel_size, stride, padding, dilation = layer.kernel_size, layer.stride, layer.padding, layer.dilation
    kernel_size = (kernel_size, kernel_size) if type(kernel_size) == int else kernel_size
    stride = (stride, stride) if type(stride) == int else stride
    padding = (padding, padding) if type(padding) == int else padding
    dilation = (dilation, dilation) if type(dilation) == int else dilation
    height_out = floor((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    width_out = floor((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return height_out, width_out
