import torch
import torch.nn.functional as F
import math
from scipy.special import comb

class BernsteinPolynomial(torch.nn.Module):
    def __init__(self, order):
        super(BernsteinPolynomial, self).__init__()
        self.order = order
        self.coeffs = torch.nn.Parameter(torch.randn(order + 1))

    def forward(self, x):
        x = x.clamp(0, 1)  # Ensure x is in [0,1] for Bernstein polynomials
        bernstein_poly = sum(
            self.coeffs[i] * comb(self.order, i) * (x ** i) * ((1 - x) ** (self.order - i))
            for i in range(self.order + 1)
        )
        return bernstein_poly

class KANLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bernstein_order=3, scale_base=1.0, base_activation=torch.nn.SiLU):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bernstein_approx = BernsteinPolynomial(bernstein_order)

        self.scale_base = scale_base
        self.base_activation = base_activation()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        bernstein_output = self.bernstein_approx(x)

        bernstein_output = bernstein_output[:, :self.out_features]  # Trim if needed
        if bernstein_output.shape[1] < base_output.shape[1]:
            padding = torch.zeros((bernstein_output.shape[0], base_output.shape[1] - bernstein_output.shape[1]), device=x.device)
            bernstein_output = torch.cat([bernstein_output, padding], dim=1)

        return base_output + bernstein_output

class KAN(torch.nn.Module):
    def __init__(self, layers_hidden, bernstein_order=3, scale_base=1.0, base_activation=torch.nn.SiLU):
        super(KAN, self).__init__()

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    bernstein_order=bernstein_order,
                    scale_base=scale_base,
                    base_activation=base_activation,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

