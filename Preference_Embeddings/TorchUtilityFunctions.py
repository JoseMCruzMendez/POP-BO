import torch

import torch

def ackley(arm: torch.Tensor) -> torch.Tensor:
    """
    u(x) = 20*exp(-0.2*sqrt(||x||_2 / n)) + exp(sum(cos(2*pi*x)) / n) - 20 - exp(1)
    supported on x ∈ [-32.768, 32.768]
    """
    n = arm.shape[0]
    # Compute the two terms of the Ackley function
    term1 = 20.0 * torch.exp(-0.2 * torch.sqrt(torch.sum(arm ** 2) / n))
    term2 = torch.exp(torch.sum(torch.cos(2.0 * torch.pi * arm)) / n)
    utility = term1 + term2 - 20.0 - torch.exp(torch.tensor(1.0, device=arm.device))
    return utility


def branin(arm: torch.Tensor) -> torch.Tensor:
    """
    u(x) = -(a*(y - b*x^2 + c*x - r)^2 + s*(1 - t)*cos(x) + s)
    supported on x ∈ [-5, 10], y ∈ [0, 15]
    """
    # Constants for the Branin function
    a = 1.0
    b = 5.1 / (4.0 * torch.pi ** 2)
    c = 5.0 / torch.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * torch.pi)

    assert arm.shape[0] == 2, "Branin requires a 2-dimensional input"
    x, y = arm[0], arm[1]
    utility = -(a * (y - b * x ** 2 + c * x - r) ** 2 + s * (1 - t) * torch.cos(x) + s)
    return utility


def eggholder(arm: torch.Tensor) -> torch.Tensor:
    """
    u(x) = (y + 47)*sin(sqrt(abs(y + x/2 + 47))) + x*sin(sqrt(abs(x - (y + 47))))
    supported on x ∈ [-512, 512], y ∈ [-512, 512]
    """
    assert arm.shape[0] == 2, "Eggholder requires a 2-dimensional input"
    x, y = arm[0], arm[1]
    term1 = (y + 47.0) * torch.sin(torch.sqrt(torch.abs(y + 0.5 * x + 47.0)))
    term2 = x * torch.sin(torch.sqrt(torch.abs(x - (y + 47.0))))
    utility = term1 + term2
    return utility


def hoelder(arm: torch.Tensor) -> torch.Tensor:
    """
    u(x) = |sin(x)*cos(y)*exp(abs(1 - sqrt(x^2 + y^2)/pi))|
    supported on x ∈ [-10, 10], y ∈ [-10, 10]
    """
    assert arm.shape[0] == 2, "Hoelder requires a 2-dimensional input"
    x, y = arm[0], arm[1]
    exponent = torch.abs(1.0 - torch.sqrt(x ** 2 + y ** 2) / torch.pi)
    utility = torch.abs(torch.sin(x) * torch.cos(y) * torch.exp(exponent))
    return utility


def matyas(arm: torch.Tensor) -> torch.Tensor:
    """
    u(x) = -(0.26*(x^2 + y^2) - 0.48*x*y)
    supported on x ∈ [-10, 10], y ∈ [-10, 10]
    """
    assert arm.shape[0] == 2, "Matyas requires a 2-dimensional input"
    x, y = arm[0], arm[1]
    utility = -(0.26 * (x ** 2 + y ** 2) - 0.48 * x * y)
    return utility


def michalewicz(arm: torch.Tensor) -> torch.Tensor:
    """
    u(x) = sum(sin(x_i)*sin(i * x_i^2 / pi)^10)
    supported on x ∈ [0, pi]
    """
    # Create a tensor [1, 2, ..., n] on the same device and dtype as arm
    indices = torch.arange(1, arm.shape[0] + 1, device=arm.device, dtype=arm.dtype)
    term = torch.sin(arm) * torch.sin(indices * (arm ** 2) / torch.pi) ** 10
    utility = torch.sum(term)
    return utility


def rosenbrock(arm: torch.Tensor) -> torch.Tensor:
    """
    u(x) = -sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    supported on x ∈ [-5, 10]
    """
    # Compute the standard Rosenbrock loss and negate it
    shifted = arm[1:]
    previous = arm[:-1]
    utility = -torch.sum(100.0 * (shifted - previous ** 2) ** 2 + (1.0 - previous) ** 2)
    return utility


def bukin(arm: torch.Tensor) -> torch.Tensor:
    """
    u(x) = -100*sqrt(abs(y - 0.01*x^2)) - 0.01*abs(x + 10)
    supported on x ∈ [-15, -5], y ∈ [-3, 3]
    """
    assert arm.shape[0] == 2, "Bukin requires a 2-dimensional input"
    x, y = arm[0], arm[1]
    term1 = -100.0 * torch.sqrt(torch.abs(y - 0.01 * x ** 2))
    term2 = -0.01 * torch.abs(x + 10.0)
    utility = term1 + term2
    return utility
