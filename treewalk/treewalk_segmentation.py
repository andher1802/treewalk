import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from skimage import io as skio
from os import walk

from skimage import filters
from skimage.color import rgb2gray, rgb2hsv

import numpy as np

from scipy import ndimage as ndi
from skimage import morphology
from skimage.morphology import square

def create_check_mask(background_mask):
	width = np.int(background_mask.shape[0]/2)
	height = np.int(background_mask.shape[1]/2)
	check_mask = np.zeros(background_mask.shape, dtype=np.int)
	check_mask[height][width] = 1
	size_check = 0.20
	for element in range(np.int(width*size_check)):
		for element2 in range(np.int(height*size_check)):
			check_mask[width+element][height+element2] = 1
			check_mask[width+element][height-element2] = 1
			check_mask[width-element][height-element2] = 1
			check_mask[width-element][height+element2] = 1
	background_checker = check_mask * background_mask
#	plt.imshow(background_checker)
#	plt.show()
	return np.sum(background_checker)

def draw_group_as_background(group, watershed_result, original_image, showImage = True):
	background_mask = (watershed_result == group)
	check_mask = create_check_mask(background_mask)
	if check_mask == 10:
		dmask = np.repeat(background_mask[...,None],3,axis=2)
#		cleaned = original_image * ~dmask
#		if showImage:
#			ax.imshow(cleaned)
#			ax.imshow(background_mask.reshape(background_mask.shape + (1,)) * np.array([1, 0, 0, 1]))
		return True, background_mask
	return False, background_mask

def segment_image(dirpath,dirpath_out,files_number=0):
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'
	plt.rcParams['figure.dpi'] = 200

	filePDDI = []
	classInput = []

	for dirpath, dirname, filename in walk(dirpath):
		filePDDI.extend(filename)

	filePDDI = [name for name in filePDDI if name.split('.')[-1] == 'jpg']
	filestocompute = []

	if files_number != 0: 
		filestocompute = list(filePDDI[:files_number])
	else:
		filestocompute = list(filePDDI)

	for filename in filestocompute:
		classInput.append(filename.split('_')[0])
		file = dirpath + filename
		img = skio.imread(file)

		img_process = rgb2hsv(img)
#		img_process = img
		channel = img_process[:,:,1]
		sobel = filters.sobel(channel)
		blurred = filters.gaussian(sobel, sigma=1.0)

		np_channel = (255*np.array(channel)).astype(int)
		# 220,30 Good results
		light_spots = np.array((np_channel > 220).nonzero()).T
		dark_spots = np.array((np_channel < 30).nonzero()).T

		bool_mask = np.zeros(channel.shape, dtype=np.bool)
		bool_mask[tuple(light_spots.T)] = True
		bool_mask[tuple(dark_spots.T)] = True
		seed_mask, num_seeds = ndi.label(bool_mask)

		#using manual seed
#		seed_mask = np.zeros(channel.shape, dtype=np.int)
#		seed_mask[0, 0] = 1 # background
#		seed_mask[int(channel.shape[0]/2), int(channel.shape[1]/2)] = 2 # foreground

		ws = morphology.watershed(blurred, seed_mask)
		background = sorted(set(ws.ravel()), key=lambda g: np.sum(ws == g), reverse=True)

#		print background
		cleaned_images = []
		N = len(background)

		print filename
		if N > 1000:
			N = 1000

		for i in range(N):
			check, mask = draw_group_as_background(background[i], ws, img_process)
			if check:
				cleaned_images.append(mask)
	
		if len(cleaned_images) > 0:
			final_mask = np.zeros(cleaned_images[0].shape, dtype=np.int)
			for masks in cleaned_images:
				final_mask += masks
			final_mask = np.invert(final_mask.astype(bool))
			final_mask = morphology.binary_erosion(final_mask,square(10))

			dmask = np.repeat(final_mask[...,None],3,axis=2)
			cleaned = img * dmask
#			plt.imshow(cleaned)
#			plt.show()
		skio.imsave(dirpath_out+filename,cleaned)

def main():
#	dirpath = "../../BaseDatos/Treewalk/Test/flower/"
	dirpath = "../../BaseDatos/oxford_classified/"
#	dirpath_out = "../../BaseDatos/Treewalk/Seg_Test/flower/"
	dirpath_out = "../../BaseDatos/oxford_segmented/"
	segment_image(dirpath,dirpath_out,0)

if __name__ == '__main__':
	main()