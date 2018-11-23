# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F


input = torch.randn(10, 10)

output = F.normalize(input, p=2, dim=1)

#dd = torch.transpose(output, 1, 1).numpy()

dd = output.numpy()

cc = output.view(-1, 1, output.size(1))

cct = torch.transpose(cc, 1, 2)

bb = torch.bmm(cc, cct).view(-1, 1)

nn = torch.ones(10, 1)

nnn = nn - bb

ab = torch.cat([nnn, bb], 1)

abb = ab.numpy()

bbb = bb.numpy()