import os
import json
import cv2
import numpy as np
import cairosvg
from plot_utils import plot_preds, svg_generate

image_base = '../../data/outdoor/cities_dataset/rgb/'
svg_base = './svg_results'

if not os.path.exists(svg_base):
    os.makedirs(svg_base)

data_filename = '../data/outdoor/cities_dataset/valid_list.txt'
with open(data_filename) as f:
    filenames = f.readlines()

filenames = filenames[50:]  # according to previous works, the testing samples are the last 350 samples of the val split
filenames = [filename.strip() for filename in filenames]
idx_to_filename = {idx: filename for idx, filename in enumerate(filenames)}

method_name = 'heat'
results_base = '../results/npy_outdoor_test_256/'

svg_method_base = os.path.join(svg_base, method_name)
if not os.path.exists(svg_method_base):
    os.makedirs(svg_method_base)

for result_filename in sorted(os.listdir(results_base)):
    file_idx = int(result_filename[:-12])
    filename = idx_to_filename[file_idx]

    image_path = os.path.join(image_base, filename + '.jpg')

    results_path = os.path.join(results_base, result_filename)
    results = np.load(results_path, allow_pickle=True).tolist()
    corners = results['corners'].astype(np.int)
    edge_ids = results['edges']
    edges = corners[edge_ids].reshape(edge_ids.shape[0], -1)

    svg = svg_generate(image_path, corners, edges, name='temp', size=256)
    svg_path = os.path.join(svg_base, 'tmp.svg')
    svg.saveas(svg_path)  # save the svg file temporarily

    svg_img_path = os.path.join(svg_method_base, '{}.png'.format(filename))
    cairosvg.svg2png(url=svg_path, write_to=svg_img_path)
