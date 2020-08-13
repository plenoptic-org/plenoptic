import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import pyrtools as pt
from plenoptic import synth
from plenoptic import simul
import plenoptic
import scipy.io as sio
import seaborn as sns

torch.__version__
numpy = lambda x: x.detach().cpu().numpy().squeeze()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

	image = plt.imread('../data/flowers.jpg').astype(float).mean(axis=2)
	image = image[0:256,0:256]

	if image.max() > 1:
		image = image/255

	im0 = torch.tensor(image,requires_grad=True,dtype = torch.float32).squeeze().unsqueeze(0).unsqueeze(0)
	

	niter = 100
	model = simul.Spectral(image.shape)
	y = model.forward(im0)
	M = synth.Metamer(im0, model)


	# set up clamps
	c = plenoptic.FourMomentsClamper(im0)

	# synthesis
	matched_image, Matched_representation = M.synthesize(max_iter=niter, learning_rate=1, 
		seed=np.random.randint(1000), clamper = c,save_iter_weights=True)


	f, ax = plt.subplots(2,3)
	sns.set()
	ax[0,0].imshow(image,cmap="gray")
	ax[0,1].imshow(matched_image, cmap="gray")
	sns.distplot(matched_image.flatten(),ax=ax[1,1])
	sns.distplot(image.flatten(),ax=ax[1,0])
	
	ax[0,2].plot(M.loss)
	sns.despine();
	plt.show()
	

	x = M.save_weights.view(256,256,niter).detach().numpy()
	




	