import torch
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch Version:", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Part 3 - Vicsek
def vicsek_fractal(iterations):
    size = 3**iterations
    if iterations == 0:
        fractal = torch.zeros(size, size)
    else:
        fractal = torch.ones(size, size)
        for i in range(3):
            for j in range(3):
                if i == 1 or j == 1:
                    third = int(size/3)
                    fractal[i*third:(i+1)*third,j*third:(j+1)*third] = vicsek_fractal(iterations-1)
    return fractal

# Generating the fractal
fractal = vicsek_fractal(5)

# Displaying the fractal
# print(fractal)
plt.imshow(fractal, cmap='gray')
plt.axis('off')
plt.show()
