import os
import json
import cv2
import numpy as np
from plot_utils import plot_preds, svg_generate
import cairosvg

image_base = '../data/cities_dataset/rgb/'
annot_base = '../data/cities_dataset/annot/'
data_filename = '../data/cities_dataset/valid_list.txt'
with open(data_filename) as f:
	filenames = f.readlines()

filenames = filenames[50:]
filenames = [filename.strip() for filename in filenames]


for filename in filenames:
	image_path = os.path.join(image_base, filename + '.jpg')
	# image = cv2.imread(image_path)
	annot_path = os.path.join(annot_base, filename + '.npy')

	annot = np.load(annot_path, allow_pickle=True, encoding='latin1').tolist()
	corners = np.array(list(annot.keys())).astype(np.int)
	
	edges = set()
	for c, others in annot.items():
		for other_c in others:
			edge = (c[0], c[1], other_c[0], other_c[1])
			edge_2 = (other_c[0], other_c[1], c[0], c[1])
			if edge not in edges and edge_2 not in edges:
				edges.add(edge)

	edges = np.array(list(edges)).astype(np.int)

	# image = plot_preds(image, corners, edges)
	# out_path = os.path.join(out_base, filename + '.png')
	# cv2.imwrite(out_path, image)

	svg = svg_generate(image_path, corners, edges, name='temp', size=256)
	svg_path = './svg_results/' + 'tmp.svg'
	svg.saveas(svg_path)	
	svg_img_path = './svg_images_256/gt/' + '{}.png'.format(filename)
	cairosvg.svg2png(url=svg_path, write_to=svg_img_path)


