import os
import numpy as np
import pickle
import cv2
from metrics.new_utils import *


class Metric():
    def calc(self, gt_data, conv_data, thresh=8.0, iou_thresh=0.7):
        ### compute corners precision/recall
        gts = gt_data['corners']
        dets = conv_data['corners']

        per_sample_corner_tp = 0.0
        per_sample_corner_fp = 0.0
        per_sample_corner_length = gts.shape[0]
        found = [False] * gts.shape[0]
        c_det_annot = {}


        # for each corner detection
        for i, det in enumerate(dets):
            # get closest gt
            near_gt = [0, 999999.0, (0.0, 0.0)]
            for k, gt in enumerate(gts):
                dist = np.linalg.norm(gt - det)
                if dist < near_gt[1]:
                    near_gt = [k, dist, gt]
            if near_gt[1] <= thresh and not found[near_gt[0]]:
                per_sample_corner_tp += 1.0
                found[near_gt[0]] = True
                c_det_annot[i] = near_gt[0]
            else:
                per_sample_corner_fp += 1.0

        per_corner_score = {
            'recall': per_sample_corner_tp / gts.shape[0],
            'precision': per_sample_corner_tp / (per_sample_corner_tp + per_sample_corner_fp + 1e-8)
        }

        ### compute edges precision/recall
        per_sample_edge_tp = 0.0
        per_sample_edge_fp = 0.0
        edge_corner_annots = gt_data['edges']
        per_sample_edge_length = edge_corner_annots.shape[0]

        false_edge_ids = []
        match_gt_ids = set()

        for l, e_det in enumerate(conv_data['edges']):
            c1, c2 = e_det

            # check if corners are mapped
            if (c1 not in c_det_annot.keys()) or (c2 not in c_det_annot.keys()):
                per_sample_edge_fp += 1.0
                false_edge_ids.append(l)
                continue
            # check hit
            c1_prime = c_det_annot[c1]
            c2_prime = c_det_annot[c2]
            is_hit = False

            for k, e_annot in enumerate(edge_corner_annots):
                c3, c4 = e_annot
                if ((c1_prime == c3) and (c2_prime == c4)) or ((c1_prime == c4) and (c2_prime == c3)):
                    is_hit = True
                    match_gt_ids.add(k)
                    break

            # hit
            if is_hit:
                per_sample_edge_tp += 1.0
            else:
                per_sample_edge_fp += 1.0
                false_edge_ids.append(l)

        per_edge_score = {
            'recall': per_sample_edge_tp / edge_corner_annots.shape[0],
            'precision': per_sample_edge_tp / (per_sample_edge_tp + per_sample_edge_fp + 1e-8)
        }

        # computer regions precision/recall
        conv_mask = render(corners=conv_data['corners'], edges=conv_data['edges'], render_pad=0, edge_linewidth=1)[0]
        conv_mask = 1 - conv_mask
        conv_mask = conv_mask.astype(np.uint8)
        labels, region_mask = cv2.connectedComponents(conv_mask, connectivity=4)

        #cv2.imwrite('mask-pred.png', region_mask.astype(np.uint8) * 20)

        background_label = region_mask[0, 0]
        all_conv_masks = []
        for region_i in range(1, labels):
            if region_i == background_label:
                continue
            the_region = region_mask == region_i
            if the_region.sum() < 20:
                continue
            all_conv_masks.append(the_region)

        gt_mask = render(corners=gt_data['corners'], edges=gt_data['edges'], render_pad=0, edge_linewidth=1)[0]
        gt_mask = 1 - gt_mask
        gt_mask = gt_mask.astype(np.uint8)
        labels, region_mask = cv2.connectedComponents(gt_mask, connectivity=4)

        #cv2.imwrite('mask-gt.png', region_mask.astype(np.uint8) * 20)

        background_label = region_mask[0, 0]
        all_gt_masks = []
        for region_i in range(1, labels):
            if region_i == background_label:
                continue
            the_region = region_mask == region_i
            if the_region.sum() < 20:
                continue
            all_gt_masks.append(the_region)

        per_sample_region_tp = 0.0
        per_sample_region_fp = 0.0
        per_sample_region_length = len(all_gt_masks)
        found = [False] * len(all_gt_masks)
        for i, r_det in enumerate(all_conv_masks):
            # gt closest gt
            near_gt = [0, 0, None]
            for k, r_gt in enumerate(all_gt_masks):
                iou = np.logical_and(r_gt, r_det).sum() / float(np.logical_or(r_gt, r_det).sum())
                if iou > near_gt[1]:
                    near_gt = [k, iou, r_gt]
            if near_gt[1] >= iou_thresh and not found[near_gt[0]]:
                per_sample_region_tp += 1.0
                found[near_gt[0]] = True
            else:
                per_sample_region_fp += 1.0

        per_region_score = {
            'recall': per_sample_region_tp / len(all_gt_masks),
            'precision': per_sample_region_tp / (per_sample_region_tp + per_sample_region_fp + 1e-8)
        }

        return {
            'corner_tp': per_sample_corner_tp,
            'corner_fp': per_sample_corner_fp,
            'corner_length': per_sample_corner_length,
            'edge_tp': per_sample_edge_tp,
            'edge_fp': per_sample_edge_fp,
            'edge_length': per_sample_edge_length,
            'region_tp': per_sample_region_tp,
            'region_fp': per_sample_region_fp,
            'region_length': per_sample_region_length,
            'corner': per_corner_score,
            'edge': per_edge_score,
            'region': per_region_score
        }


def compute_metrics(gt_data, pred_data):
    metric = Metric()
    score = metric.calc(gt_data, pred_data)
    return score


def get_recall_and_precision(tp, fp, length):
    recall = tp / (length + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    return recall, precision


if __name__ == '__main__':
    base_path = './'
    gt_datapath = '../data/cities_dataset/annot'
    metric = Metric()
    corner_tp = 0.0
    corner_fp = 0.0
    corner_length = 0.0
    edge_tp = 0.0
    edge_fp = 0.0
    edge_length = 0.0
    region_tp = 0.0
    region_fp = 0.0
    region_length = 0.0
    for file_name in os.listdir(base_path):
        if len(file_name) < 10:
            continue
        f = open(os.path.join(base_path, file_name), 'rb')
        gt_data = np.load(os.path.join(gt_datapath, file_name + '.npy'), allow_pickle=True).tolist()
        candidate = pickle.load(f)
        conv_corners = candidate.graph.getCornersArray()
        conv_edges = candidate.graph.getEdgesArray()
        conv_data = {'corners': conv_corners, 'edges': conv_edges}
        score = metric.calc(gt_data, conv_data)
        corner_tp += score['corner_tp']
        corner_fp += score['corner_fp']
        corner_length += score['corner_length']
        edge_tp += score['edge_tp']
        edge_fp += score['edge_fp']
        edge_length += score['edge_length']
        region_tp += score['region_tp']
        region_fp += score['region_fp']
        region_length += score['region_length']

    f = open(os.path.join(base_path, 'score.txt'), 'w')
    # corner
    recall, precision = get_recall_and_precision(corner_tp, corner_fp, corner_length)
    f_score = 2.0 * precision * recall / (recall + precision + 1e-8)
    print('corners - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))
    f.write('corners - precision: %.3f recall: %.3f f_score: %.3f\n' % (precision, recall, f_score))

    # edge
    recall, precision = get_recall_and_precision(edge_tp, edge_fp, edge_length)
    f_score = 2.0 * precision * recall / (recall + precision + 1e-8)
    print('edges - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))
    f.write('edges - precision: %.3f recall: %.3f f_score: %.3f\n' % (precision, recall, f_score))

    # region
    recall, precision = get_recall_and_precision(region_tp, region_fp, region_length)
    f_score = 2.0 * precision * recall / (recall + precision + 1e-8)
    print('regions - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))
    f.write('regions - precision: %.3f recall: %.3f f_score: %.3f\n' % (precision, recall, f_score))

    f.close()
