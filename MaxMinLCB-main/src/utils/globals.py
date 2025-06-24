from src.utils.kernels import LinearKernel, RBFKernel, Matern12Kernel, Matern32Kernel, Matern53Kernel


KERNELS = {
    "linear": LinearKernel,
    "rbf": RBFKernel,
    "matern12": Matern12Kernel,
    "matern32": Matern32Kernel,
    "matern53": Matern53Kernel,
}