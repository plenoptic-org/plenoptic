import plenoptic as po
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import pyrtools as pt
import itertools
import pandas as pd
import matplotlib as mpl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
to_numpy = lambda x : x.detach().cpu().numpy().squeeze()

image = plt.imread('../data/metal.pgm').astype(float)
print(image.shape)
im0 = torch.tensor(image,requires_grad=True,dtype = torch.float32).squeeze().unsqueeze(0).unsqueeze(0)
x = po.make_basic_stimuli()
x = x[0,0,:,:].unsqueeze(0).unsqueeze(0)
print(x.shape)
model = po.simulate.Portilla_Simoncelli(x.shape[-2:],n_scales=4,n_orientations=4,Na=9)
y = model.forward(x)