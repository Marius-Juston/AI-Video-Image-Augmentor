import torch
from torch.autograd import Function
from torch.nn.modules.module import Module

if torch.cuda.is_available():
    import correlation_cuda
else:
    import dain_model.misc.PWCNet.correlation_package_pytorch1_0.correlation_cpu as correlation_cuda


def create_CorrelationFunction(pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
    class CorrelationFunction(Function):
        @staticmethod
        def forward(self, input1, input2):
            self.save_for_backward(input1, input2)

            with torch.cuda.device_of(input1):
                rbot1 = input1.new()
                rbot2 = input2.new()
                output = input1.new()

                correlation_cuda.forward(input1, input2, rbot1, rbot2, output,
                                         pad_size, kernel_size, max_displacement, stride1, stride2,
                                         corr_multiply)

            return output

        @staticmethod
        def backward(self, grad_output):
            input1, input2 = self.saved_tensors

            with torch.cuda.device_of(input1):
                rbot1 = input1.new()
                rbot2 = input2.new()

                grad_input1 = input1.new()
                grad_input2 = input2.new()

                correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                                          pad_size, kernel_size, max_displacement, stride1,
                                          stride2, corr_multiply)

            return grad_input1, grad_input2

    return CorrelationFunction()


class Correlation(Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        result = create_CorrelationFunction(self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2,
                                     self.corr_multiply).apply(input1, input2)

        return result
