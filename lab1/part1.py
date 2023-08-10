import torch
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch Version:", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Part 1
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]
x = torch.Tensor(X)
y = torch.Tensor(Y)
x = x.to(device)
y = y.to(device)

# Gaussian function
sd = 2.0    # standard Deviation
gaussian = torch.exp(-(x**2+y**2)/sd)
# 2D sine function: y(t) = A*sin(2*PI*f*t+p)
A = 1.0 # amplitude
f = 1.0 # ordinary frequency
p = 0   # phase
sine_2d = A*torch.sin(2*torch.pi*f*(x+y)+p)

z = gaussian * sine_2d

plt.imshow(z.cpu().numpy())
plt.tight_layout()
plt.show()
# print(type(X), type(Y), type(z), type(z.numpy()))
# print(X, Y, x, y, z, sep="\n\n")

