import torch
import numpy as np
import scipy.ndimage.filters as filters
import cv2
import itertools

NEIGHBOUR_SIZE = 5
MATCH_THRESH = 5
LOCAL_MAX_THRESH = 0.01
viz_count = 0

# pre-compute all combinations to generate edge candidates faster
all_combibations = dict()
for length in range(2, 351):
    ids = np.arange(length)
    combs = np.array(list(itertools.combinations(ids, 2)))
    all_combibations[length] = combs


def prepare_edge_data(c_outputs, annots, images, max_corner_num):
    bs = c_outputs.shape[0]
    # prepares parameters for each sample of the batch
    all_results = list()

    for b_i in range(bs):
        annot = annots[b_i]
        output = c_outputs[b_i]
        results = process_each_sample({'annot': annot, 'output': output, 'viz_img': images[b_i]}, max_corner_num)
        all_results.append(results)

    processed_corners = [item['corners'] for item in all_results]
    edge_coords = [item['edges'] for item in all_results]
    edge_labels = [item['labels'] for item in all_results]

    edge_info = {
        'edge_coords': edge_coords,
        'edge_labels': edge_labels,
        'processed_corners': processed_corners
    }

    edge_data = collate_edge_info(edge_info)
    return edge_data


def process_annot(annot, do_round=True):
    corners = np.array(list(annot.keys()))
    ind = np.lexsort(corners.T)  # sort the g.t. corners to fix the order for the matching later
    corners = corners[ind]  # sorted by y, then x
    corner_mapping = {tuple(k): v for v, k in enumerate(corners)}

    edges = list()
    for c, connections in annot.items():
        for other_c in connections:
            edge_pair = (corner_mapping[c], corner_mapping[tuple(other_c)])
            edges.append(edge_pair)
    corner_degrees = [len(annot[tuple(c)]) for c in corners]
    if do_round:
        corners = corners.round()
    return corners, edges, corner_degrees


def process_each_sample(data, max_corner_num):
    annot = data['annot']
    output = data['output']

    preds = output.detach().cpu().numpy()

    data_max = filters.maximum_filter(preds, NEIGHBOUR_SIZE)
    maxima = (preds == data_max)
    data_min = filters.minimum_filter(preds, NEIGHBOUR_SIZE)
    diff = ((data_max - data_min) > 0)
    maxima[diff == 0] = 0
    local_maximas = np.where((maxima > 0) & (preds > LOCAL_MAX_THRESH))
    pred_corners = np.stack(local_maximas, axis=-1)[:, [1, 0]]  # to (x, y format)

    # produce edge labels labels from pred corners here

    processed_corners, edges, labels = get_edge_label_mix_gt(pred_corners, annot, max_corner_num)
    # global viz_count
    # viz_img = data['viz_img']
    #output_path = './viz_training/{}_example_gt.png'.format(viz_count)
    #_visualize_edge_training_data(processed_corners, edges, labels, viz_img, output_path)
    #viz_count += 1

    results = {
        'corners': processed_corners,
        'edges': edges,
        'labels': labels,
    }
    return results


def get_edge_label_mix_gt(pred_corners, annot, max_corner_num):
    ind = np.lexsort(pred_corners.T)  # sort the pred corners to fix the order for matching
    pred_corners = pred_corners[ind]  # sorted by y, then x
    gt_corners, edge_pairs, corner_degrees = process_annot(annot)

    output_to_gt = dict()
    gt_to_output = dict()
    diff = np.sqrt(((pred_corners[:, None] - gt_corners) ** 2).sum(-1))
    diff = diff.T

    if len(pred_corners) > 0:
        for target_i, target in enumerate(gt_corners):
            dist = diff[target_i]
            if len(output_to_gt) > 0:
                dist[list(output_to_gt.keys())] = 1000  # ignore already matched pred corners
            min_dist = dist.min()
            min_idx = dist.argmin()
            if min_dist < MATCH_THRESH and min_idx not in output_to_gt:  # a positive match
                output_to_gt[min_idx] = (target_i, min_dist)
                gt_to_output[target_i] = min_idx

    all_corners = gt_corners.copy()

    # replace matched g.t. corners with pred corners
    for gt_i in range(len(gt_corners)):
       if gt_i in gt_to_output:
            all_corners[gt_i] = pred_corners[gt_to_output[gt_i]]

    nm_pred_ids = [i for i in range(len(pred_corners)) if i not in output_to_gt]
    nm_pred_ids = np.random.permutation(nm_pred_ids)
    if len(nm_pred_ids) > 0:
        nm_pred_corners = pred_corners[nm_pred_ids]
        #if len(nm_pred_ids) + len(all_corners) <= 150:
        if len(nm_pred_ids) + len(all_corners) <= max_corner_num:
            all_corners = np.concatenate([all_corners, nm_pred_corners], axis=0)
        else:
            #all_corners = np.concatenate([all_corners, nm_pred_corners[:(150 - len(gt_corners)), :]], axis=0)
            all_corners = np.concatenate([all_corners, nm_pred_corners[:(max_corner_num - len(gt_corners)), :]], axis=0)

    processed_corners, edges, edge_ids, labels = _get_edges(all_corners, edge_pairs)

    return processed_corners, edges, labels


def _get_edges(corners, edge_pairs):
    ind = np.lexsort(corners.T)
    corners = corners[ind]  # sorted by y, then x
    corners = corners.round()
    id_mapping = {old: new for new, old in enumerate(ind)}

    all_ids = all_combibations[len(corners)]
    edges = corners[all_ids]
    labels = np.zeros(edges.shape[0])

    N = len(corners)
    edge_pairs = [(id_mapping[p[0]], id_mapping[p[1]]) for p in edge_pairs]
    edge_pairs = [p for p in edge_pairs if p[0] < p[1]]
    pos_ids = [int((2 * N - 1 - p[0]) * p[0] / 2 + p[1] - p[0] - 1) for p in edge_pairs]
    labels[pos_ids] = 1

    edge_ids = np.array(all_ids)
    return corners, edges, edge_ids, labels


def collate_edge_info(data):
    batched_data = {}
    lengths_info = {}
    for field in data.keys():
        batch_values = data[field]
        all_lens = [len(value) for value in batch_values]
        max_len = max(all_lens)
        pad_value = 0
        batch_values = [pad_sequence(value, max_len, pad_value) for value in batch_values]
        batch_values = np.stack(batch_values, axis=0)

        if field in ['edge_coords', 'edge_labels', 'gt_values']:
            batch_values = torch.Tensor(batch_values).long()
        if field in ['processed_corners', 'edge_coords']:
            lengths_info[field] = all_lens
        batched_data[field] = batch_values

    # Add length and mask into the data, the mask if for Transformers' input format, True means padding
    for field, lengths in lengths_info.items():
        lengths_str = field + '_lengths'
        batched_data[lengths_str] = torch.Tensor(lengths).long()
        mask = torch.arange(max(lengths))
        mask = mask.unsqueeze(0).repeat(batched_data[field].shape[0], 1)
        mask = mask >= batched_data[lengths_str].unsqueeze(-1)
        mask_str = field + '_mask'
        batched_data[mask_str] = mask

    return batched_data


def pad_sequence(seq, length, pad_value=0):
    if len(seq) == length:
        return seq
    else:
        pad_len = length - len(seq)
        if len(seq.shape) == 1:
            if pad_value == 0:
                paddings = np.zeros([pad_len, ])
            else:
                paddings = np.ones([pad_len, ]) * pad_value
        else:
            if pad_value == 0:
                paddings = np.zeros([pad_len, ] + list(seq.shape[1:]))
            else:
                paddings = np.ones([pad_len, ] + list(seq.shape[1:])) * pad_value
        padded_seq = np.concatenate([seq, paddings], axis=0)
        return padded_seq


def get_infer_edge_pairs(corners, confs):
    ind = np.lexsort(corners.T)
    corners = corners[ind]  # sorted by y, then x
    confs = confs[ind]

    edge_ids = all_combibations[len(corners)]
    edge_coords = corners[edge_ids]

    edge_coords = torch.tensor(np.array(edge_coords)).unsqueeze(0).long()
    mask = torch.zeros([edge_coords.shape[0], edge_coords.shape[1]]).bool()
    edge_ids = torch.tensor(np.array(edge_ids))
    return corners, confs, edge_coords, mask, edge_ids


def _visualize_edge_training_data(corners, edges, edge_labels, image, save_path):
    image = image.transpose([1, 2, 0])
    image = (image * 255).astype(np.uint8)
    image = np.ascontiguousarray(image)

    for edge, label in zip(edges, edge_labels):
        if label == 1:
            cv2.line(image, tuple(edge[0].astype(np.int)), tuple(edge[1].astype(np.int)), (255, 255, 0), 2)

    for c in corners:
        cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 0, 255), -1)

    cv2.imwrite(save_path, image)
