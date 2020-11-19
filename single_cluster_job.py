

if __name__ == "__main__": 
	import sys
	import numpy as np
	from VAE_class import VAE
	
	X_train = np.load('imagenet.npy')
	print('loaded set!')
	print(X_train.shape)

	beta = sys.argv[1]
	path = '{}{}'.format('trained_models/trained_vae_beta_val_',beta)
	vae = VAE((32, 32, 3),1024, beta)
	vae.fit(X_train, 12, True, path)
