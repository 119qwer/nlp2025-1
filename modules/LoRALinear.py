import torch.nn.functional as F
from torch import nn
import torch
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, lora_alpha=16, lora_dropout=0.0, bias=True):
        super().__init__()
        self.r = r
        self.scaling = lora_alpha / r

        # 기존 weight는 freeze (동결)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False

        # LoRA 파라미터
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)

        if self.r > 0:
            lora_intermediate = F.linear(self.dropout(x), self.lora_A)
            lora = F.linear(lora_intermediate, self.lora_B)
            return base + self.scaling * lora

        return base
