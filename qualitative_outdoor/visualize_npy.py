import os
import json
import cv2
import numpy as np
import cairosvg
from plot_utils import plot_preds, svg_generate

image_base = '../../data/outdoor/cities_dataset/rgb/'

data_filename = '../data/outdoor/cities_dataset/valid_list.txt'
with open(data_filename) as f:
    filenames = f.readlines()

filenames = filenames[50:] # according to previous works, the testing samples are the last 350 samples of the val split
filenames = [filename.strip() for filename in filenames]
idx_to_filename = {idx:filename for idx, filename in enumerate(filenames)}

# results_base = './letr_npy_results/stage_focal_query_100/'
results_base = '../results/npy_outdoor_test_256/'
# results_base = './hawp_npy_results/hawp_npy_256'
# results_base = './convmpn_npy'
#results_base = './exp_cls_npy'

for result_filename in sorted(os.listdir(results_base)):
    file_idx = int(result_filename[:-12])
    filename = idx_to_filename[file_idx]

    #filename = result_filename[:-4]

    image_path = os.path.join(image_base, filename + '.jpg')

    results_path = os.path.join(results_base, result_filename)
    results = np.load(results_path, allow_pickle=True).tolist()
    corners = results['corners'].astype(np.int)
    edge_ids = results['edges']
    edges = corners[edge_ids].reshape(edge_ids.shape[0], -1)

    # image = plot_preds(image, corners, edges)
    # out_path = os.path.join(out_base, filename + '.png')
    # cv2.imwrite(out_path, image)
	
    svg = svg_generate(image_path, corners, edges, name='temp', size=256)
    svg_path = os.path.join('./svg_results', 'tmp.svg')
    svg.saveas(svg_path)	
    svg_img_path = './svg_results/heat/' + '{}.png'.format(filename)
    cairosvg.svg2png(url=svg_path, write_to=svg_img_path)


