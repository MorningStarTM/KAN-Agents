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
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:] 
                
            )

        assert bases.size() == (x.size(0), self.input_dim, self.grid_size + self.spline_order)
        return bases.contiguous()
    

    def curve2coeff(self, x:torch.Tensor, y:torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.input_dim
        assert y.size() == (x.size(0), self.input_dim, self.output_dim)

        A = self.b_splines(x).transpose(0,1)
        B = y.transpose(0,1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2,0,1)

        assert result.size() == (
            self.output_dim,
            self.input_dim,
            self.grid_size + self.spline_order
        )
        return result.contiguous()
    

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)
    
    def forward(self, x:torch.Tensor):
        assert x.size(-1) == self.input_dim
        original_shape = x.shape
        x = x.view(-1, self.input_dim)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.output_dim, -1)

        )
        output = base_output + spline_output
        output = output.view(*original_shape[:-1], self.output_dim)
        return output
    

    @torch.no_grad()
    def update_grid(self, x:torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.input_dim
        batch = x.size(0)

        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch-1, self.grid_size+1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0]+2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size+1, dtype=torch.float32, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin
        )


        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))



    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )