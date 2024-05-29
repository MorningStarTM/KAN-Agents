import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


class KANNeuron(nn.Module):
    def __init__(self, input_dim, output_dim, grid_size=5, spline_order=3, scale_noise=0.1, 
                 scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, base_activate=torch.nn.SiLU,
                 grid_eps=0.02, grid_range=[-1,1]):
        super(KANNeuron, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size

        self.grid = ((torch.arange(-spline_order, grid_size+spline_order+1)*h+grid_range[0]).expand(input_dim, -1).conti)

        self.register_buffer("grid", self.grid)

        self.base_weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.spline_weight = nn.Parameter(torch.Tensor(output_dim, input_dim, grid_size+spline_order))

        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(output_dim, input_dim))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activate()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size+1, self.input_dim, self.output_dim) - 1/2
                
                )*self.scale_noise / self.grid_size
            
            )

            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * 
                self.curv2coeff( self.grid.T[self.spline_order : -self.spline_order],
                                    noise,
                                                          )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    
    def b_splines(self, x:torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.input_dim

        grid: torch.Tensor = (
            self.grid
        )  

        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order+1):
            bases = (
                
            )