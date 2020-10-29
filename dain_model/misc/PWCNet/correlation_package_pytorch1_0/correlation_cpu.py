import numpy as np

from .correlation_cuda_kernel import correlation_forward_cuda_kernel
import torch


def forward(input1, input2, rInput1, rInput2, output,
            pad_size,
            kernel_size,
            max_displacement,
            stride1,
            stride2,
            corr_type_multiply):
    batchSize = input1.shape[0]

    nInputChannels = input1.shape[1]
    inputHeight = input1.shape[2]
    inputWidth = input1.shape[3]

    kernel_radius = (kernel_size - 1) / 2
    border_radius = kernel_radius + max_displacement

    paddedInputHeight = inputHeight + 2 * pad_size
    paddedInputWidth = inputWidth + 2 * pad_size

    nOutputChannels = ((max_displacement / stride2) * 2 + 1) * ((max_displacement / stride2) * 2 + 1)

    outputHeight = np.ceil((paddedInputHeight - 2 * border_radius) / (stride1))
    outputwidth = np.ceil((paddedInputWidth - 2 * border_radius) / (stride1))

    nOutputChannels = int(nOutputChannels)
    outputHeight = int(outputHeight)
    outputwidth = int(outputwidth)

    rInput1 = torch.zeros((batchSize, paddedInputHeight, paddedInputWidth, nInputChannels))
    rInput2 = torch.zeros((batchSize, paddedInputHeight, paddedInputWidth, nInputChannels))
    output = torch.zeros((batchSize, nOutputChannels, outputHeight, outputwidth))

    success = correlation_forward_cuda_kernel(
        output,
        output.shape[0],
        output.shape[1],
        output.shape[2],
        output.shape[3],
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        input1,
        input1.shape[1],
        input1.shape[2],
        input1.shape[3],
        input1.stride(0),
        input1.stride(1),
        input1.stride(2),
        input1.stride(3),
        input2,
        input2.shape[1],
        input2.stride(0),
        input2.stride(1),
        input2.stride(2),
        input2.stride(3),
        rInput1,
        rInput2,
        pad_size,
        kernel_size,
        max_displacement,
        stride1,
        stride2,
        corr_type_multiply

    )

    print(success)

    return success
