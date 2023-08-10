import torch
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch Version:", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Part 2
zoom_rate = 0
offset = 0, 0
set0 = -1.3, 1.3, -2, 1, 0.005
set1 = -0.65, 0.65, -1, 0.5, 0.0025
# set2 = -0.4, 0.2, -0.5, 0.2, 0.004
set_julia = -2,2,-2,2,0.001
ymn, ymx, xmn, xmx, spc = set_julia
Y, X = np.mgrid[ymn:ymx:spc, xmn:xmx:spc]
x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x,y) #important!
zs = z
ns = torch.zeros_like(z)

z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

for i in range(64): 
    # Mandelbrot Set: new z = z^2+z
    zs_ = zs*zs + z
    # Julia Set: new z = z^2+c
    # zs_ = zs*zs + (-0.54+0.54j)
    not_diverged = torch.abs(zs_) < 4.0
    ns += not_diverged.type(torch.FloatTensor) 
    zs = zs_

fig = plt.figure(figsize=(16,10)) 
def processFractal(a): 
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1]) 
    img = np.concatenate([10+20*np.cos(a_cyclic), 
                          30+50*np.sin(a_cyclic), 
                          155-80*np.cos(a_cyclic)], 2) 
    img[a==a.max()] = 0 
    a=img 
    a=np.uint8(np.clip(a,0,255)) 
    return a 
plt.imshow(processFractal(ns.numpy())) 
plt.tight_layout(pad=0) 
plt.show()

# print(X, Y, sep='\n\n')
# X, Y= np.mgrid[-2:1:0.005, -1.3:1.3:0.005]
# print(X, Y, sep='\n\n')
