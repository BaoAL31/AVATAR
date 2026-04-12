import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_in = self.dropout(x)
        lora_out = (lora_in @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * lora_out


class _LoRAConvBase(nn.Module):
    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LoRAConv1d(_LoRAConvBase):
    def __init__(self, base: nn.Conv1d, rank: int, alpha: float, dropout: float):
        super().__init__(base, rank, alpha, dropout)
        self.down = nn.Conv1d(base.in_channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv1d(
            rank,
            base.out_channels,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=1,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.up(self.dropout(self.down(x)))


class LoRAConv2d(_LoRAConvBase):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float, dropout: float):
        super().__init__(base, rank, alpha, dropout)
        self.down = nn.Conv2d(base.in_channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(
            rank,
            base.out_channels,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=1,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.up(self.dropout(self.down(x)))


class LoRAConv3d(_LoRAConvBase):
    def __init__(self, base: nn.Conv3d, rank: int, alpha: float, dropout: float):
        super().__init__(base, rank, alpha, dropout)
        self.down = nn.Conv3d(base.in_channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv3d(
            rank,
            base.out_channels,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=1,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.up(self.dropout(self.down(x)))


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    if "." in module_name:
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = root.get_submodule(parent_name)
    else:
        parent = root
        child_name = module_name
    setattr(parent, child_name, new_module)


def _name_matches(name: str, patterns: Iterable[str]) -> bool:
    return any(pattern in name for pattern in patterns)


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    linear_patterns: Iterable[str],
    conv_patterns: Iterable[str],
) -> Dict[str, int]:
    replaced = {"linear": 0, "conv1d": 0, "conv2d": 0, "conv3d": 0}
    named_modules: Tuple[Tuple[str, nn.Module], ...] = tuple(model.named_modules())

    for name, module in named_modules:
        if not name:
            continue
        if isinstance(module, (LoRALinear, LoRAConv1d, LoRAConv2d, LoRAConv3d)):
            continue

        if isinstance(module, nn.Linear) and _name_matches(name, linear_patterns):
            _replace_module(model, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
            replaced["linear"] += 1
        elif isinstance(module, nn.Conv1d) and _name_matches(name, conv_patterns):
            _replace_module(model, name, LoRAConv1d(module, rank=rank, alpha=alpha, dropout=dropout))
            replaced["conv1d"] += 1
        elif isinstance(module, nn.Conv2d) and _name_matches(name, conv_patterns):
            _replace_module(model, name, LoRAConv2d(module, rank=rank, alpha=alpha, dropout=dropout))
            replaced["conv2d"] += 1
        elif isinstance(module, nn.Conv3d) and _name_matches(name, conv_patterns):
            _replace_module(model, name, LoRAConv3d(module, rank=rank, alpha=alpha, dropout=dropout))
            replaced["conv3d"] += 1

    return replaced
