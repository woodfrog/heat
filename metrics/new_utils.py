import numpy as np
import matplotlib.pyplot as plt
import cv2
import threading
import os
import skimage
import random
import time

TWO_CORNER_MINIMUM_DISTANCE = 5
SAFE_NUM = 3
score_weights = (1., 2., 100.)


#########################################################################################
################################# General Functions #####################################
#########################################################################################
def swap_two_corner_place(corners, edges, id1, id2):
    for edge_i in range(edges.shape[0]):
        if edges[edge_i, 0] == id1:
            edges[edge_i, 0] = id2
        elif edges[edge_i, 0] == id2:
            edges[edge_i, 0] = id1
        if edges[edge_i, 1] == id1:
            edges[edge_i, 1] = id2
        elif edges[edge_i, 1] == id2:
            edges[edge_i, 1] = id1
    temp = corners[id1].copy()
    corners[id1] = corners[id2]
    corners[id2] = temp
    return corners, edges


def get_neighbor_corner_id(corner_id, edges):
    where = np.where(edges == corner_id)
    return edges[where[0], 1 - where[1]]


def swap_two_edge_place(edges, id1, id2):
    temp = edges[id1].copy()
    edges[id1] = edges[id2]
    edges[id2] = temp
    return edges


def degree_of_three_corners(cornerA, cornerB, cornerM):
    # cornerM is middle corner
    AM_length = l2_distance(cornerA, cornerM)
    BM_length = l2_distance(cornerB, cornerM)
    dot = np.dot((cornerA[0] - cornerM[0], cornerA[1] - cornerM[1]),
                 (cornerB[0] - cornerM[0], cornerB[1] - cornerM[1]))
    cos = dot / (AM_length + 1e-8) / (BM_length + 1e-8)
    cos = min(1, max(-1, cos))
    degree = np.arccos(cos)
    return degree / np.pi * 180


def sort_graph(corners, edges):
    corners = corners.copy()
    edges = edges.copy()
    for corner_i in range(corners.shape[0]):
        min_id = -1
        min_pos = corners[corner_i]
        for corner_j in range(corner_i + 1, corners.shape[0]):
            if (corners[corner_j, 0] < min_pos[0]) or \
                    (corners[corner_j, 0] == min_pos[0] and corners[corner_j, 1] < min_pos[1]):
                min_pos = corners[corner_j]
                min_id = corner_j
        if min_id != -1:
            corners, edges = swap_two_corner_place(corners, edges, corner_i, min_id)

    for edge_i in range(edges.shape[0]):
        if edges[edge_i, 0] > edges[edge_i, 1]:
            temp = edges[edge_i, 0]
            edges[edge_i, 0] = edges[edge_i, 1]
            edges[edge_i, 1] = temp

    for edge_i in range(edges.shape[0]):
        min_id = -1
        min_pos = edges[edge_i]
        for edge_j in range(edge_i + 1, edges.shape[0]):
            if (edges[edge_j, 0] < min_pos[0]) or \
                    (edges[edge_j, 0] == min_pos[0] and edges[edge_j, 1] < min_pos[1]):
                min_pos = edges[edge_j]
                min_id = edge_j
        if min_id != -1:
            edges = swap_two_edge_place(edges, edge_i, min_id)

    return corners, edges


def IOU(maskA, maskB):
    return np.logical_and(maskA, maskB).sum() / np.logical_or(maskA, maskB).sum()


def render(corners, edges, render_pad=0, edge_linewidth=2, corner_size=3, scale=1.):
    size = int(256 * scale)
    mask = np.ones((2, size, size)) * render_pad

    corners = np.round(corners.copy() * scale).astype(np.int)
    for edge_i in range(edges.shape[0]):
        a = edges[edge_i, 0]
        b = edges[edge_i, 1]
        mask[0] = cv2.line(mask[0], (int(corners[a, 1]), int(corners[a, 0])),
                           (int(corners[b, 1]), int(corners[b, 0])), 1.0, thickness=edge_linewidth)
    for corner_i in range(corners.shape[0]):
        mask[1] = cv2.circle(mask[1], (int(corners[corner_i, 1]), int(corners[corner_i, 0])), corner_size, 1.0, -1)

    return mask


def patch_samples(edge_num, batch_size):
    num = edge_num // batch_size
    patchs = []
    for i in range(num):
        patchs.append([i * batch_size + j for j in range(batch_size)])

    if edge_num % batch_size != 0:
        patchs.append([j for j in range(batch_size * num, edge_num)])

    return patchs


def l2_distance(x1, x2):
    return np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)


def triangle_region(A, B, C):
    l1 = np.linalg.norm(np.array(A) - np.array(B))
    l2 = np.linalg.norm(np.array(A) - np.array(C))
    l3 = np.linalg.norm(np.array(B) - np.array(C))
    p = (l1 + l2 + l3) / 2
    area = np.sqrt(np.abs(p * (p - l1) * (p - l2) * (p - l3)))
    return area


def remove_intersection_and_duplicate(corners, edges, name):
    over_all_flag = False
    ori_corners = corners.copy()
    ori_edges = edges.copy()
    while True:
        flag = False
        for edge_i in range(edges.shape[0]):
            for edge_j in range(edge_i + 1, edges.shape[0]):
                corner11 = corners[edges[edge_i, 0]]
                corner12 = corners[edges[edge_i, 1]]
                corner21 = corners[edges[edge_j, 0]]
                corner22 = corners[edges[edge_j, 1]]

                y1 = corner11[0]
                x1 = corner11[1]
                y2 = corner12[0]
                x2 = corner12[1]
                a1 = y1 - y2
                b1 = x2 - x1
                c1 = x1 * y2 - x2 * y1
                flag1 = (a1 * corner21[1] + b1 * corner21[0] + c1) * (a1 * corner22[1] + b1 * corner22[0] + c1)

                y1 = corner21[0]
                x1 = corner21[1]
                y2 = corner22[0]
                x2 = corner22[1]
                a2 = y1 - y2
                b2 = x2 - x1
                c2 = x1 * y2 - x2 * y1
                flag2 = (a2 * corner11[1] + b2 * corner11[0] + c2) * (a2 * corner12[1] + b2 * corner12[0] + c2)

                if flag1 < -1e-5 and flag2 < -1e-5:
                    # intersection!
                    over_all_flag = True
                    flag = True

                    new_x = (c2 * b1 - c1 * b2) / (a1 * b2 - a2 * b1)
                    new_y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)

                    temp_d = 3
                    temp_id = -1
                    if l2_distance((new_y, new_x), corner11) < temp_d:
                        temp_id = edges[edge_i, 0]
                        temp_d = l2_distance((new_y, new_x), corner11)
                    if l2_distance((new_y, new_x), corner12) < temp_d:
                        temp_id = edges[edge_i, 1]
                        temp_d = l2_distance((new_y, new_x), corner12)
                    if l2_distance((new_y, new_x), corner21) < temp_d:
                        temp_id = edges[edge_j, 0]
                        temp_d = l2_distance((new_y, new_x), corner21)
                    if l2_distance((new_y, new_x), corner22) < temp_d:
                        temp_id = edges[edge_j, 1]
                        temp_d = l2_distance((new_y, new_x), corner22)
                    if temp_id != -1:
                        if edges[edge_i, 0] != temp_id and edges[edge_i, 1] != temp_id:
                            tt = edges[edge_i, 0]
                            edges[edge_i, 0] = temp_id
                            edges = np.append(edges, np.array([(temp_id, tt)]), 0)
                        if edges[edge_j, 0] != temp_id and edges[edge_j, 1] != temp_id:
                            tt = edges[edge_j, 0]
                            edges[edge_j, 0] = temp_id
                            edges = np.append(edges, np.array([(temp_id, tt)]), 0)
                    else:
                        corners = np.append(corners, np.array([(new_y, new_x)]), 0)
                        edge_id1 = edges[edge_i, 1]
                        edge_id2 = edges[edge_j, 1]
                        edges[edge_i, 1] = corners.shape[0] - 1
                        edges[edge_j, 1] = corners.shape[0] - 1
                        edges = np.append(edges, np.array([(edge_id1, corners.shape[0] - 1)]), 0)
                        edges = np.append(edges, np.array([(edge_id2, corners.shape[0] - 1)]), 0)
                    break
            if flag:
                break
        if flag:
            continue
        break

    # remove duplicate and zero degree
    graph = Graph(np.round(corners), edges)
    for corner_i in reversed(range(len(graph.getCorners()))):
        corner_ele1 = graph.getCorners()[corner_i]
        for corner_j in reversed(range(corner_i)):
            corner_ele2 = graph.getCorners()[corner_j]
            if l2_distance(corner_ele1.x, corner_ele2.x) < 3:
                connected_edge = graph.getEdgeConnected(corner_ele1)
                for edge_ele in connected_edge:
                    if edge_ele.x[0] == corner_ele1:
                        another = edge_ele.x[1]
                    else:
                        another = edge_ele.x[0]
                    if another == corner_ele2:
                        graph.remove(edge_ele)
                    edge_ele.x = (another, corner_ele2)
                graph.remove(corner_ele1)
    for corner_ele in graph.getCorners():
        if graph.getCornerDegree(corner_ele) == 0:
            graph.remove(corner_ele)

    corners = graph.getCornersArray()
    edges = graph.getEdgesArray()
    # if over_all_flag:
    #    plt.subplot(121)
    #    ori = render(ori_corners, ori_edges, edge_linewidth=1, corner_size=1)
    #    temp = np.concatenate((ori.transpose((1,2,0)), np.zeros((ori.shape[1],ori.shape[2],1))),2)
    #    plt.imshow(temp)
    #    plt.subplot(122)
    #    new_ = render(corners, edges, edge_linewidth=1, corner_size=1)
    #    temp = np.concatenate((new_.transpose((1,2,0)), np.zeros((new_.shape[1],new_.shape[2],1))),2)
    #    plt.imshow(temp)
    #    plt.show()

    return corners, edges


def get_two_edge_intersection_location(corner11, corner12, corner21, corner22):
    y1 = corner11[0]
    x1 = corner11[1]
    y2 = corner12[0]
    x2 = corner12[1]
    a1 = y1 - y2
    b1 = x2 - x1
    c1 = x1 * y2 - x2 * y1

    y1 = corner21[0]
    x1 = corner21[1]
    y2 = corner22[0]
    x2 = corner22[1]
    a2 = y1 - y2
    b2 = x2 - x1
    c2 = x1 * y2 - x2 * y1

    l = a1 * b2 - a2 * b1
    if l == 0:
        l = 1e-5

    new_x = (c2 * b1 - c1 * b2) / l
    new_y = (a2 * c1 - a1 * c2) / l

    return round(new_y), round(new_x)


def get_distance_of_corner_and_edge(corner1, corner2, corner):
    x = corner[0]
    y = corner[1]
    x1 = corner1[0]
    y1 = corner1[1]
    x2 = corner2[0]
    y2 = corner2[1]

    cross = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1)
    if cross <= 0:
        # dist to corner1
        return np.linalg.norm((x - x1, y - y1))

    d2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
    if cross >= d2:
        # dist to corner2
        return np.linalg.norm((x - x2, y - y2))

    r = cross / d2
    px = x1 + (x2 - x1) * r
    py = y1 + (y2 - y1) * r
    return np.linalg.norm((x - px, y - py))


#########################################################################################
################################# Dataset Functions #####################################
#########################################################################################
def EuclideanDistance(A, B):
    BT = B.transpose()
    vecProd = np.dot(A, BT)

    SqA = A ** 2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED


def samedirection(conv_corner_id, gt_corner_id, conv_corners, gt_corners, conv_edges, gt_edges):
    # degree
    if np.where(conv_edges == conv_corner_id)[0].shape[0] != np.where(gt_edges == gt_corner_id)[0].shape[0]:
        return False

    # direction
    place = np.where(conv_edges == conv_corner_id)
    neighbor_id = conv_edges[place[0], 1 - place[1]]

    distance = conv_corners[conv_corner_id] - conv_corners[neighbor_id]
    direction = np.arctan2(distance[:, 0], distance[:, 1]) * 180 / np.pi / 15
    direction = (direction + 24) % 24

    conv_dir = np.sort(direction)

    place = np.where(gt_edges == gt_corner_id)
    neighbor_id = gt_edges[place[0], 1 - place[1]]

    distance = gt_corners[gt_corner_id] - gt_corners[neighbor_id]
    direction = np.arctan2(distance[:, 0], distance[:, 1]) * 180 / np.pi / 15
    direction = (direction + 24) % 24

    gt_dir = np.sort(direction)

    conv_dir = list(conv_dir)
    gt_dir = list(gt_dir)
    for angle in gt_dir:
        temp = sorted(conv_dir, key=lambda x: min(np.abs(x - angle), 24 - np.abs(x - angle)))
        if min(np.abs(temp[0] - angle), 24 - np.abs(temp[0] - angle)) <= 1.3:
            conv_dir.remove(temp[0])
        else:
            return False
    return True


def simplify_gt(gt_match_location, gt_corner, gt_edge):
    graph = Graph(np.round(gt_corner), gt_edge)
    for idx, corner in enumerate(graph.getCorners()):
        # use score to store the matching info
        corner.store_score(gt_match_location[idx])

    for idx, corner in enumerate(graph.getCorners()):
        if corner.get_score() is None:
            connected_edges = graph.getEdgeConnected(corner)
            neighbor_corners = []
            for edge in connected_edges:
                if edge.x[0] != corner:
                    neighbor_corners.append(edge.x[0])
                    continue
                if edge.x[1] != corner:
                    neighbor_corners.append(edge.x[1])
                    continue
                raise BaseException()
            neighbor_corners = sorted(neighbor_corners, key=lambda ele: l2_distance(ele.x, corner.x))
            for neighbor_ele in neighbor_corners:
                if l2_distance(neighbor_ele.x, corner.x) > 8:
                    break
                if neighbor_ele.get_score() is None:
                    continue
                # find the suitable neighbor that replace corner
                for ele in neighbor_corners:
                    if ele == neighbor_ele:
                        continue
                    graph.add_edge(ele, neighbor_ele)
                neighbor_ele.x = (0.7 * neighbor_ele.x[0] + 0.3 * corner.x[0],
                                  0.7 * neighbor_ele.x[1] + 0.3 * corner.x[1])
                graph.remove(corner)
                break
    return graph.getCornersArray(), graph.getEdgesArray()


def get_wrong_corners(corners, gt_corners, edges, gt_edges):
    corners = corners.copy()
    gt_corners = gt_corners.copy()
    edges = edges.copy()
    gt_edges = gt_edges.copy()
    dist_matrix = EuclideanDistance(gt_corners, corners)
    assigned_id = set()
    gt_match_same_degree = []
    gt_match_location = []
    for gt_i in range(gt_corners.shape[0]):
        sort_id = np.argsort(dist_matrix[gt_i]).__array__()[0]
        flag = True
        for id_ in sort_id:
            if dist_matrix[gt_i, id_] > 7:
                break
            temete = samedirection(id_, gt_i, corners, gt_corners, edges, gt_edges)
            if temete == False:
                break
            elif id_ not in assigned_id:
                assigned_id.add(id_)
                gt_match_same_degree.append(id_)
                flag = False
                break
        if flag:
            gt_match_same_degree.append(None)

    matched = []
    gt_match_location = [None for _ in range(gt_corners.shape[0])]
    for gt_i in sorted(list(range(gt_corners.shape[0])), key=lambda i: np.min(dist_matrix[i])):
        sort_id = np.argsort(dist_matrix[gt_i]).__array__()[0]
        if dist_matrix[gt_i, sort_id[0]] > 7:
            gt_match_location[gt_i] = None
        else:
            for c_i in sort_id:
                if c_i in matched:
                    continue
                if dist_matrix[gt_i, c_i] > 7:
                    gt_match_location[gt_i] = None
                    break
                else:
                    gt_match_location[gt_i] = c_i
                    matched.append(c_i)
                    break

    return set(range(corners.shape[0])) - assigned_id, gt_match_same_degree, gt_match_location


def get_wrong_edges(corners, gt_corners, edges, gt_edges, gt_match):
    edges = edges.copy()
    gt_edges = gt_edges.copy()

    all_possible_good_edges = []
    for edge_i in range(gt_edges.shape[0]):
        if gt_match[gt_edges[edge_i, 0]] is None or gt_match[gt_edges[edge_i, 1]] is None:
            continue
        all_possible_good_edges.append((gt_match[gt_edges[edge_i, 0]], gt_match[gt_edges[edge_i, 1]]))
    false_edge_id = []
    for edge_i in range(edges.shape[0]):
        id1 = edges[edge_i][0]
        id2 = edges[edge_i][1]
        if (id1, id2) not in all_possible_good_edges and (id2, id1) not in all_possible_good_edges:
            false_edge_id.append(edge_i)
            continue

    return false_edge_id


def get_corner_bin_map(corners, corner_list_for_each_bin, bin_size=10):
    bin_map = np.zeros((bin_size, 256, 256))
    for bin_i in range(bin_size):
        bin_map[bin_i] = render(corners[corner_list_for_each_bin[bin_i]], np.array([]), render_pad=0)[1]
    return bin_map


#########################################################################################
################################ Searching Functions ####################################
#########################################################################################
def visualization(candidate, show=True):
    corners = candidate.graph.getCornersArray()
    edges = candidate.graph.getEdgesArray()
    mask = render(corners, edges)
    mask = np.transpose(np.concatenate((mask, np.zeros((1, 256, 256))), 0), (1, 2, 0))
    plt.imshow(mask)
    if show:
        plt.show()


def check_intersection(edge1, edge2):
    corner11 = edge1.x[0].x
    corner12 = edge1.x[1].x
    corner21 = edge2.x[0].x
    corner22 = edge2.x[1].x

    y1 = corner11[0]
    x1 = corner11[1]
    y2 = corner12[0]
    x2 = corner12[1]
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    flag1 = (a * corner21[1] + b * corner21[0] + c) * (a * corner22[1] + b * corner22[0] + c)

    y1 = corner21[0]
    x1 = corner21[1]
    y2 = corner22[0]
    x2 = corner22[1]
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    flag2 = (a * corner11[1] + b * corner11[0] + c) * (a * corner12[1] + b * corner12[0] + c)

    if flag1 < -1e-6 and flag2 < -1e-6:
        return True

    return False


def adding_a_corner_by_triangle_operation(candidate):
    new_candidates = []
    name = candidate.name
    gt_mask = region_cache.get_region(name)
    gt_mask = gt_mask > 0.4
    gt_mask_grow = cv2.dilate(gt_mask.astype(np.float64), np.ones((3, 3), np.uint8), iterations=6) > 0

    # get the current candidate region mask
    conv_mask = render(corners=candidate.graph.getCornersArray(), edges=candidate.graph.getEdgesArray(),
                       render_pad=0, edge_linewidth=1)[0]
    conv_mask = 1 - conv_mask
    conv_mask = conv_mask.astype(np.uint8)
    labels, region_mask = cv2.connectedComponents(conv_mask, connectivity=4)

    background_label = region_mask[0, 0]
    all_masks = []
    for region_i in range(1, labels):
        if region_i == background_label:
            continue
        the_region = region_mask == region_i
        if the_region.sum() < 20:
            continue
        all_masks.append(the_region)

    candidate_mask = (np.sum(all_masks, 0) + (1 - conv_mask)) > 0

    final_mask = np.logical_xor(gt_mask_grow, np.logical_and(candidate_mask, gt_mask_grow))

    for corner_i in range(random.randint(0, 16), 256, 16):
        for corner_j in range(random.randint(0, 16), 256, 16):
            if candidate.addable((corner_i, corner_j)):
                if final_mask[corner_i, corner_j] == True:  # inside the region
                    new_corner = Element((corner_i, corner_j))
                    new_candidate = candidate.generate_new_candidate_add_a_corner(new_corner)
                    new_graph = new_candidate.graph
                    corners = new_graph.getCorners()

                    # find two suitable existed corners to make into a triangle (no intersection and no colinear)
                    for id_A in range(len(corners)):
                        ele_A = corners[id_A]
                        if ele_A == new_corner:
                            continue
                        for id_B in range(id_A + 1, len(corners)):
                            ele_B = corners[id_B]
                            if ele_B == new_corner:
                                continue
                            if new_graph.has_edge(new_corner, ele_A) is not None:
                                raise BaseException('should not have edge in this case')
                            if new_graph.has_edge(new_corner, ele_B) is not None:
                                raise BaseException('should not have edge in this case')
                            temp_edge1 = Element((new_corner, ele_A))
                            temp_edge2 = Element((new_corner, ele_B))

                            # check if addable
                            if new_candidate.addable(temp_edge1) is False:
                                continue
                            if new_candidate.addable(temp_edge2) is False:
                                continue

                            # avoid intersection
                            if new_graph.checkIntersectionEdge(temp_edge1):
                                continue
                            if new_graph.checkIntersectionEdge(temp_edge2):
                                continue

                            # avoid too small triangle
                            if triangle_region(new_corner.x, ele_A.x, ele_B.x) < 20:
                                continue

                            ### avoid colinear edge (only when fold case)
                            # for edge1
                            neighbor_edges = new_graph.getEdgeConnected(temp_edge1)
                            flag_ = True
                            for neighbor in neighbor_edges:
                                if new_corner in neighbor.x:
                                    raise BaseException('new corner should not in any edge')
                                elif ele_A in neighbor.x:
                                    shared_corner = ele_A
                                else:
                                    raise BaseException('error.')
                                two_neighbor = {neighbor.x[0], neighbor.x[1], ele_A, new_corner}
                                two_neighbor.remove(shared_corner)
                                assert len(two_neighbor) == 2
                                two_neighbor = tuple(two_neighbor)

                                line1 = np.array(shared_corner.x) - np.array(two_neighbor[0].x)
                                line2 = np.array(shared_corner.x) - np.array(two_neighbor[1].x)
                                cos = np.dot(line1, line2) / (np.linalg.norm(line1) * np.linalg.norm(line2))
                                cos = min(1, max(-1, cos))
                                if np.arccos(cos) < np.pi / 9:  # 20 degree
                                    flag_ = False
                                    break
                            if flag_ is False:
                                continue
                            # for edge2
                            neighbor_edges = new_graph.getEdgeConnected(temp_edge2)
                            flag_ = True
                            for neighbor in neighbor_edges:
                                if new_corner in neighbor.x:
                                    raise BaseException('new corner should not in any edge')
                                elif ele_B in neighbor.x:
                                    shared_corner = ele_B
                                else:
                                    raise BaseException('error.')
                                two_neighbor = {neighbor.x[0], neighbor.x[1], ele_B, new_corner}
                                two_neighbor.remove(shared_corner)
                                assert len(two_neighbor) == 2
                                two_neighbor = tuple(two_neighbor)

                                line1 = np.array(shared_corner.x) - np.array(two_neighbor[0].x)
                                line2 = np.array(shared_corner.x) - np.array(two_neighbor[1].x)
                                cos = np.dot(line1, line2) / (np.linalg.norm(line1) * np.linalg.norm(line2))
                                cos = min(1, max(-1, cos))
                                if np.arccos(cos) < np.pi / 9:  # 20 degree
                                    flag_ = False
                                    break
                            if flag_ is False:
                                continue

                            # make new candidate
                            try:
                                new_ = new_candidate.generate_new_candidate_add_an_edge(new_corner, ele_A)
                                new_ = new_.generate_new_candidate_add_an_edge(new_corner, ele_B)
                                new_candidates.append(new_)
                            except:
                                continue
                            # plt.subplot(151)
                            # visualization(candidate, show=False)
                            # plt.subplot(152)
                            # plt.imshow(final_mask)
                            # plt.subplot(153)
                            # plt.imshow(candidate_mask)
                            # plt.subplot(154)
                            # plt.imshow(gt_mask_grow)
                            # plt.subplot(155)
                            # visualization(new_, show=False)
                            # plt.show()

    return new_candidates


def adding_an_edge_from_new_corner_operation(candidate):
    new_candidates = []
    name = candidate.name
    gt_mask = region_cache.get_region(name)
    gt_mask = gt_mask > 0.4
    gt_mask_grow = cv2.dilate(gt_mask.astype(np.float64), np.ones((3, 3), np.uint8), iterations=6) > 0

    # get the current candidate region mask
    conv_mask = render(corners=candidate.graph.getCornersArray(), edges=candidate.graph.getEdgesArray(),
                       render_pad=0, edge_linewidth=1)[0]
    conv_mask = 1 - conv_mask
    conv_mask = conv_mask.astype(np.uint8)
    labels, region_mask = cv2.connectedComponents(conv_mask, connectivity=4)
    background_label = region_mask[0, 0]
    all_masks = []
    for region_i in range(1, labels):
        if region_i == background_label:
            continue
        the_region = region_mask == region_i
        if the_region.sum() < 20:
            continue
        all_masks.append(the_region)
    candidate_mask = (np.sum(all_masks, 0) + (1 - conv_mask)) > 0

    final_mask = np.logical_xor(gt_mask_grow, np.logical_and(candidate_mask, gt_mask_grow))
    for corner_i in range(random.randint(0, 16), 256, 16):
        for corner_j in range(random.randint(0, 16), 256, 16):
            if candidate.addable((corner_i, corner_j)):
                if final_mask[corner_i, corner_j] == True:
                    # inside the region
                    new_corner = Element((corner_i, corner_j))
                    new_candidate = candidate.generate_new_candidate_add_a_corner(new_corner)
                    new_graph = new_candidate.graph
                    corners = new_graph.getCorners()

                    # find a suitable existed corner that can make
                    # a new edge with new_corner (no intersection and colinear)
                    for corner_ele in corners:
                        if corner_ele == new_corner:
                            continue
                        if new_graph.has_edge(new_corner, corner_ele) is not None:
                            raise BaseException('should not have edge in this case')
                        temp_edge = Element((new_corner, corner_ele))

                        # check if addable
                        if new_candidate.addable(temp_edge) is False:
                            continue

                        # avoid intersection
                        if new_graph.checkIntersectionEdge(temp_edge):
                            continue

                        # avoid colinear edge
                        neighbor_edges = new_graph.getEdgeConnected(temp_edge)
                        flag_ = True
                        for neighbor in neighbor_edges:
                            if new_corner in neighbor.x:
                                raise BaseException('new corner should not in any edge')
                            elif corner_ele in neighbor.x:
                                shared_corner = corner_ele
                            else:
                                raise BaseException('error.')
                            two_neighbor = {neighbor.x[0], neighbor.x[1], corner_ele, new_corner}
                            two_neighbor.remove(shared_corner)
                            assert len(two_neighbor) == 2
                            two_neighbor = tuple(two_neighbor)

                            line1 = np.array(shared_corner.x) - np.array(two_neighbor[0].x)
                            line2 = np.array(shared_corner.x) - np.array(two_neighbor[1].x)
                            cos = np.dot(line1, line2) / (np.linalg.norm(line1) * np.linalg.norm(line2))
                            cos = min(1, max(-1, cos))
                            if np.arccos(cos) < np.pi / 9:  # 20 degree
                                flag_ = False
                                break
                        if flag_ is False:
                            continue

                        # make new candidate
                        try:
                            new_ = new_candidate.generate_new_candidate_add_an_edge(new_corner, corner_ele)
                            new_candidates.append(new_)
                        except:
                            continue

    return new_candidates


def removing_a_corner_operation(candidate):
    new_candidates = []
    graph = candidate.graph
    corners = graph.getCorners()
    for the_corner in corners:
        if candidate.removable(the_corner):
            try:
                new_ = candidate.generate_new_candidate_remove_a_corner(the_corner)
                new_candidates.append(new_)
            except:
                continue

    return new_candidates


def removing_a_colinear_corner_operation(candidate):
    new_candidates = []
    graph = candidate.graph
    corners = graph.getCorners()
    for the_corner in corners:
        if candidate.removable(the_corner):  # NO NEED TO CHECK IF COLINEAR and graph.checkColinearCorner(the_corner):
            try:
                new_ = candidate.generate_new_candidate_remove_a_colinear_corner(the_corner)

                if new_.graph.checkIntersectionEdge():
                    continue
                new_candidates.append(new_)
            except:
                continue

    return new_candidates


def adding_an_edge_operation(candidate):
    new_candidates = []
    graph = candidate.graph
    corners = graph.getCorners()
    for corner_i in range(len(corners)):
        cornerA = corners[corner_i]
        for corner_j in range(corner_i + 1, len(corners)):
            cornerB = corners[corner_j]
            if graph.has_edge(cornerA, cornerB) is not None:
                continue

            temp_edge = Element((cornerA, cornerB))
            # check if addable (not in existed_before dict)
            if candidate.addable(temp_edge) is False:
                continue

            if graph.checkIntersectionEdge(temp_edge):
                continue

            # avoid adding a colinear edge
            neighbor_edges = graph.getEdgeConnected(temp_edge)
            flag_ = True
            for neighbor in neighbor_edges:
                if cornerA in neighbor.x:
                    shared_corner = cornerA
                elif cornerB in neighbor.x:
                    shared_corner = cornerB
                else:
                    raise BaseException('error.')
                two_neighbor = {neighbor.x[0], neighbor.x[1], cornerA, cornerB}
                two_neighbor.remove(shared_corner)
                assert len(two_neighbor) == 2
                two_neighbor = tuple(two_neighbor)

                line1 = np.array(shared_corner.x) - np.array(two_neighbor[0].x)
                line2 = np.array(two_neighbor[1].x) - np.array(shared_corner.x)
                cos = np.dot(line1, line2) / (np.linalg.norm(line1) * np.linalg.norm(line2))
                cos = min(1, max(-1, cos))
                if np.arccos(cos) < np.pi / 18 or np.arccos(cos) > np.pi - np.pi / 18:  # 10 degree
                    flag_ = False
                    break
            if flag_ is False:
                continue

            # make new candidate
            try:
                new_ = candidate.generate_new_candidate_add_an_edge(cornerA, cornerB)
                new_candidates.append(new_)
            except:
                continue

    return new_candidates


def removing_an_edge_operation(candidate):
    new_candidates = []
    graph = candidate.graph
    edges = graph.getEdges()
    for edge_ele in edges:
        if candidate.removable(edge_ele):
            try:
                new_ = candidate.generate_new_candidate_remove_an_edge(edge_ele)
                new_candidates.append(new_)
            except:
                continue

    return new_candidates


def adding_an_edge_from_gt(candidate, gt_data):
    new_candidates = []
    corners_array = candidate.graph.getCornersArray()
    edges_array = candidate.graph.getEdgesArray()

    gt_corners = gt_data['corners'].copy()
    gt_edges = gt_data['edges'].copy()

    _, _, map_same_location = get_wrong_corners(
        corners_array, gt_corners, edges_array, gt_edges)

    gt_corners, gt_edges = simplify_gt(map_same_location, gt_corners, gt_edges)

    _, _, map_same_location = get_wrong_corners(
        corners_array, gt_corners, edges_array, gt_edges)

    for corner_i in range(gt_corners.shape[0]):
        if map_same_location[corner_i] is None:
            # doesn't exist in candidate
            neighbor_id = get_neighbor_corner_id(corner_i, gt_edges)
            for corner_j in neighbor_id:
                if map_same_location[corner_j] is not None:
                    # exist corner in candidate that maps neighbor corner
                    new_candidate = candidate.copy()
                    new_corner = Element(
                        (
                            int(np.round(gt_corners[corner_i, 0])), int(np.round(gt_corners[corner_i, 1]))
                        )
                    )
                    if new_candidate.addable(new_corner) is False:
                        continue
                    # new corner can be too close to an edge
                    flag = False
                    for edge_ele in new_candidate.graph.getEdges():
                        if get_distance_of_corner_and_edge(edge_ele.x[0].x, edge_ele.x[1].x, new_corner.x) < 7:
                            flag = True
                            break
                    if flag:
                        continue

                    new_corner = new_candidate.addCorner(new_corner)
                    neighbor_index = map_same_location[corner_j]
                    neighbor_corner = new_candidate.graph.getCorners()[neighbor_index]
                    new_edge = new_candidate.addEdge(new_corner, neighbor_corner)
                    if new_candidate.graph.checkIntersectionEdge(new_edge):
                        continue
                    new_candidates.append(new_candidate)

    return new_candidates


def adding_a_corner_from_two_edges_extension(candidate):
    new_candidates = []
    graph = candidate.graph
    edges = candidate.graph.getEdges()
    for edge_i in range(len(edges)):
        for edge_j in range(edge_i + 1, len(edges)):
            edgeA = edges[edge_i]
            edgeB = edges[edge_j]
            if graph.isNeighbor(edgeA, edgeB):
                continue
            intersection_loc = get_two_edge_intersection_location(edgeA.x[0].x, edgeA.x[1].x, edgeB.x[0].x,
                                                                  edgeB.x[1].x)
            if intersection_loc[0] >= 255 or intersection_loc[1] >= 255 or \
                    intersection_loc[0] <= 0 or intersection_loc[1] <= 0:
                continue
            # intersection point can not be too close to an edge
            flag = False
            for edge_ele in graph.getEdges():
                if get_distance_of_corner_and_edge(edge_ele.x[0].x, edge_ele.x[1].x, intersection_loc) < 7:
                    flag = True
                    break
            if flag:
                continue
            new_candidate = candidate.copy()
            new_graph = new_candidate.graph
            new_edgeA = new_graph.getRealElement(edgeA)
            new_edgeB = new_graph.getRealElement(edgeB)
            new_corner = Element(intersection_loc)
            if new_candidate.addable(new_corner) is False:
                continue
            new_corner = new_candidate.addCorner_v2(new_corner)
            # get cornerA and cornerB from edgeA, edgeB
            if l2_distance(new_corner.x, new_edgeA.x[0].x) < l2_distance(new_corner.x, new_edgeA.x[1].x):
                cornerA = new_edgeA.x[0]
            else:
                cornerA = new_edgeA.x[1]
            if l2_distance(new_corner.x, new_edgeB.x[0].x) < l2_distance(new_corner.x, new_edgeB.x[1].x):
                cornerB = new_edgeB.x[0]
            else:
                cornerB = new_edgeB.x[1]

            # new edge can not be too short
            if l2_distance(cornerA.x, new_corner.x) < 7:
                continue
            if l2_distance(cornerB.x, new_corner.x) < 7:
                continue

            # new intersection cannot be too flat
            if degree_of_three_corners(cornerA.x, cornerB.x, new_corner.x) > 165:
                continue

            flag = False
            for edge_ele in new_graph.getEdges():
                if new_corner in edge_ele.x and cornerA in edge_ele.x:
                    flag = True
                    break
                if edge_ele.x[0] not in (new_corner, cornerA):
                    l = get_distance_of_corner_and_edge(new_corner.x, cornerA.x, edge_ele.x[0].x)
                    if l <= 7:
                        flag = True
                        break
                if edge_ele.x[1] not in (new_corner, cornerA):
                    l = get_distance_of_corner_and_edge(new_corner.x, cornerA.x, edge_ele.x[1].x)
                    if l <= 7:
                        flag = True
                        break
            if flag:
                continue
            add_edgeA = new_candidate.addEdge(new_corner, cornerA)
            if new_graph.checkIntersectionEdge(add_edgeA):
                continue

            flag = False
            for edge_ele in new_graph.getEdges():
                if new_corner in edge_ele.x and cornerB in edge_ele.x:
                    flag = True
                    break
                if edge_ele.x[0] not in (new_corner, cornerB):
                    l = get_distance_of_corner_and_edge(new_corner.x, cornerB.x, edge_ele.x[0].x)
                    if l <= 7:
                        flag = True
                        break
                if edge_ele.x[1] not in (new_corner, cornerB):
                    l = get_distance_of_corner_and_edge(new_corner.x, cornerB.x, edge_ele.x[1].x)
                    if l <= 7:
                        flag = True
                        break
            if flag:
                continue
            add_edgeB = new_candidate.addEdge(new_corner, cornerB)
            if new_graph.checkIntersectionEdge(add_edgeB):
                continue

            # make real new candidate
            # new_candidate = candidate.copy()
            # new_graph = new_candidate.graph
            # new_corner = Element(intersection_loc)
            # new_corner = new_graph.add_corner_v2(new_corner)
            # new_candidate = new_candidate.generate_new_candidate_add_an_edge(new_corner, cornerA)
            # new_candidate = new_candidate.generate_new_candidate_add_an_edge(new_corner, cornerB)

            new_candidates.append(new_candidate)
    return new_candidates


def adding_a_corner_from_parallel(candidate):
    new_candidates = []
    graph = candidate.graph
    edges = candidate.graph.getEdges()
    for edge_i in range(len(edges)):
        for edge_j in range(edge_i + 1, len(edges)):
            edgeA = edges[edge_i]
            edgeB = edges[edge_j]
            # get intersection loc
            if graph.isNeighbor(edgeA, edgeB):
                shared_corner = edgeA.x[0] if edgeA.x[0] in edgeB.x else edgeA.x[1]
                intersection_loc = shared_corner.x
            else:
                intersection_loc = get_two_edge_intersection_location(
                    edgeA.x[0].x, edgeA.x[1].x, edgeB.x[0].x, edgeB.x[1].x)
            if intersection_loc[0] >= 255 or intersection_loc[1] >= 255 or \
                    intersection_loc[0] <= 0 or intersection_loc[1] <= 0:
                continue

            # get another two loc
            locA = edgeA.x[1].x if \
                l2_distance(edgeA.x[0].x, intersection_loc) < l2_distance(edgeA.x[1].x, intersection_loc) else \
                edgeA.x[0].x
            locB = edgeB.x[1].x if \
                l2_distance(edgeB.x[0].x, intersection_loc) < l2_distance(edgeB.x[1].x, intersection_loc) else \
                edgeB.x[0].x

            # get new loc
            new_loc = (locA[0] + locB[0] - intersection_loc[0], locA[1] + locB[1] - intersection_loc[1])
            if new_loc[0] >= 255 or new_loc[1] >= 255 or \
                    new_loc[0] <= 0 or new_loc[1] <= 0:
                continue

            new_corner = Element(new_loc)
            new_candidate = candidate.copy()
            new_graph = new_candidate.graph
            edgeA = new_graph.getRealElement(edgeA)
            edgeB = new_graph.getRealElement(edgeB)
            if new_candidate.addable(new_corner) is False:
                continue
            new_corner = new_candidate.addCorner_v2(new_corner)
            # get cornerA and cornerB from edgeA, edgeB
            cornerA = edgeA.x[1] if l2_distance(edgeA.x[0].x, intersection_loc) < l2_distance(edgeA.x[1].x,
                                                                                              intersection_loc) \
                else edgeA.x[0]
            cornerB = edgeB.x[1] if l2_distance(edgeB.x[0].x, intersection_loc) < l2_distance(edgeB.x[1].x,
                                                                                              intersection_loc) \
                else edgeB.x[0]

            # new edge can not be too short
            if l2_distance(cornerA.x, new_corner.x) < 12:
                continue
            if l2_distance(cornerB.x, new_corner.x) < 12:
                continue

            flag = False
            for edge_ele in new_graph.getEdges():
                if new_corner in edge_ele.x and cornerA in edge_ele.x:
                    flag = True
                    break
                if edge_ele.x[0] not in (new_corner, cornerA):
                    l = get_distance_of_corner_and_edge(new_corner.x, cornerA.x, edge_ele.x[0].x)
                    if l <= 7:
                        flag = True
                        break
                if edge_ele.x[1] not in (new_corner, cornerA):
                    l = get_distance_of_corner_and_edge(new_corner.x, cornerA.x, edge_ele.x[1].x)
                    if l <= 7:
                        flag = True
                        break
            if flag:
                continue
            add_edgeA = new_candidate.addEdge(new_corner, cornerA)
            if new_graph.checkIntersectionEdge(add_edgeA):
                continue

            flag = False
            for edge_ele in new_graph.getEdges():
                if new_corner in edge_ele.x and cornerB in edge_ele.x:
                    flag = True
                    break
                if edge_ele.x[0] not in (new_corner, cornerB):
                    l = get_distance_of_corner_and_edge(new_corner.x, cornerB.x, edge_ele.x[0].x)
                    if l <= 7:
                        flag = True
                        break
                if edge_ele.x[1] not in (new_corner, cornerB):
                    l = get_distance_of_corner_and_edge(new_corner.x, cornerB.x, edge_ele.x[1].x)
                    if l <= 7:
                        flag = True
                        break
            if flag:
                continue
            add_edgeB = new_candidate.addEdge(new_corner, cornerB)
            if new_graph.checkIntersectionEdge(add_edgeB):
                continue

            new_candidates.append(new_candidate)
    return new_candidates


def adding_a_orthogonal_edge(candidate):
    new_candidates = []
    graph = candidate.graph
    edges = candidate.graph.getEdges()
    for edge in edges:
        cornerA = edge.x[0]
        cornerB = edge.x[1]

        # get orthogonal direction
        dir_ = (cornerA.x[1] - cornerB.x[1], cornerB.x[0] - cornerA.x[0])

        for the_corner in edge.x:
            temp_orth_loc = (the_corner.x[0] - dir_[0], the_corner.x[1] - dir_[1])
            for inter_edge in edges:
                if inter_edge == edge:
                    continue
                if the_corner in inter_edge.x:
                    continue
                intersection_loc = get_two_edge_intersection_location(
                    the_corner.x, temp_orth_loc, inter_edge.x[0].x, inter_edge.x[1].x
                )
                if intersection_loc[0] >= 255 or intersection_loc[1] >= 255 or \
                        intersection_loc[0] <= 0 or intersection_loc[1] <= 0:
                    continue
                if np.dot((inter_edge.x[0].x[0] - intersection_loc[0], inter_edge.x[0].x[1] - intersection_loc[1]),
                          (inter_edge.x[1].x[0] - intersection_loc[0], inter_edge.x[1].x[1] - intersection_loc[1])) > 0:
                    # which means the intersection is not inside inter_edge but at the edge extension
                    continue
                if l2_distance(intersection_loc, inter_edge.x[0].x) < 5 or \
                        l2_distance(intersection_loc, inter_edge.x[1].x) < 5:
                    continue

                # no thin degree with neighbor edge
                flag = False
                neighbor_corners = graph.getNeighborCorner(the_corner)
                for corner_ele in neighbor_corners:
                    if corner_ele in edge.x:
                        continue
                    if degree_of_three_corners(corner_ele.x, intersection_loc, the_corner.x) < 15:
                        flag = True
                        break
                    if degree_of_three_corners(corner_ele.x, intersection_loc, the_corner.x) > 165:
                        flag = True
                        break
                if flag:
                    continue

                new_candidate = candidate.copy()
                new_graph = new_candidate.graph
                new_corner = Element(intersection_loc)
                if new_candidate.addable(new_corner) is False:
                    continue
                new_corner = new_candidate.addCorner_v2(new_corner)

                # new edge can not be too short
                if l2_distance(new_corner.x, the_corner.x) < 7:
                    continue

                add_edge = new_candidate.addEdge(new_corner, new_graph.getRealElement(the_corner))
                if new_graph.checkIntersectionEdge(add_edge):
                    continue

                new_candidates.append(new_candidate)
    return new_candidates


class _thread(threading.Thread):
    def __init__(self, threadID, name, candidate, lock, result_list, func):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.candidate = candidate
        self.lock = lock
        self.result_list = result_list
        self.func = func

    def run(self):
        print('running id: ', self.name)
        start_time = time.time()
        candidates = self.func(self.candidate)
        print('test: =================================', self.name, len(candidates))
        self.lock.acquire()
        self.result_list.extend(candidates)
        self.lock.release()
        print(self.name, "spend time: {}s".format(time.time() - start_time))


def candidate_enumerate_training(candidate, gt):
    new_candidates = []
    # remove a corner
    try:
        new_ = removing_a_corner_operation(candidate)
        if len(new_) > 0:
            new_candidates.append(random.choice(new_))
    except:
        print('something wrong with remove a corner !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    # remove a colinear corner
    try:
        new_ = removing_a_colinear_corner_operation(candidate)
        if len(new_) > 0:
            new_candidates.append(random.choice(new_))
    except:
        print('something wrong with remove a colinear corner !!!!!!!!!!!!!!!!!!!!!!!!!!!')

    # remove an edge
    try:
        new_ = removing_an_edge_operation(candidate)
        if len(new_) > 0:
            new_candidates.append(random.choice(new_))
    except:
        print('something wrong with remove an edge !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    # add an edge from existed corner
    try:
        new_ = adding_an_edge_operation(candidate)
        if len(new_) > 0:
            new_candidates.append(random.choice(new_))
    except:
        print('something wrong with add an edge from existed corner !!!!!!!!!!!!!!!!!!!!')

    # add a corner from two edges
    try:
        new_ = adding_a_corner_from_two_edges_extension(candidate)
        if len(new_) > 0:
            new_candidates.append(random.choice(new_))
    except:
        print('something wrong with add a corner from two edges !!!!!!!!!!!!!!!!!!!!!!!!')

    try:
        new_ = adding_a_corner_from_parallel(candidate)
        if len(new_) > 0:
            new_candidates.append(random.choice(new_))
    except:
        print('something wrong with add a corner from parallel !!!!!!!!!!!!!!!!!!!!!!!!')

    # add an edge from gt
    try:
        new_ = adding_an_edge_from_gt(candidate, gt)
        if len(new_) > 0:
            new_candidates.append(random.choice(new_))
    except:
        print('something wrong with add an edge from gt !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    # add a orthogonal edge
    try:
        new_ = adding_a_orthogonal_edge(candidate)
        if len(new_) > 0:
            new_candidates.append(random.choice(new_))
    except:
        print('something wrong with add a orthogonal edge !!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    return new_candidates


def candidate_enumerate(candidate):
    new_candidates = []
    new_candidates.extend(removing_a_corner_operation(candidate))
    new_candidates.extend(removing_a_colinear_corner_operation(candidate))
    new_candidates.extend(removing_an_edge_operation(candidate))
    new_candidates.extend(adding_an_edge_operation(candidate))
    new_candidates.extend(adding_a_corner_from_two_edges_extension(candidate))
    new_candidates.extend(adding_a_corner_from_parallel(candidate))
    new_candidates.extend(adding_a_orthogonal_edge(candidate))

    return new_candidates


def candidate_enumerate_thread(candidate):
    new_candidates = []
    lock = threading.Lock()

    thread1 = _thread(1, 'remove_a_corner', candidate, lock, new_candidates, removing_a_corner_operation)
    thread2 = _thread(2, 'remove_a_colinear_corner', candidate, lock, new_candidates,
                      removing_a_colinear_corner_operation)
    thread3 = _thread(3, 'add_an_edge', candidate, lock, new_candidates, adding_an_edge_operation)
    thread4 = _thread(4, 'remove_an_edge', candidate, lock, new_candidates, removing_an_edge_operation)

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()

    threads = []
    threads.append(thread1)
    threads.append(thread2)
    threads.append(thread3)
    threads.append(thread4)

    for t in threads:
        t.join()

    return new_candidates


def reduce_duplicate_candidate(candidates):
    i = 0
    while i < len(candidates):
        for j in reversed(range(i + 1, len(candidates))):
            if candidates[i].equal(candidates[j]):
                del candidates[j]
        i = i + 1
    return candidates


def save_candidate_image(candidate, base_path, base_name):
    corners = candidate.graph.getCornersArray()
    edges = candidate.graph.getEdgesArray()
    # graph svg
    svg = svg_generate(corners, edges, base_name, samecolor=True)
    svg.saveas(os.path.join(base_path, base_name + '.svg'))
    # corner image
    temp_mask = np.zeros((256, 256))
    for ele in candidate.graph.getCorners():
        if ele.get_score() < 0:
            temp_mask = cv2.circle(temp_mask, ele.x[::-1], 3, 1, -1)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(1, 1)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(temp_mask, aspect='auto')
    fig.savefig(os.path.join(base_path, base_name + '_corner.png'), dpi=256)
    # edges image
    temp_mask = np.zeros((256, 256))
    for ele in candidate.graph.getEdges():
        if ele.get_score() < 0:
            A = ele.x[0]
            B = ele.x[1]
            temp_mask = cv2.line(temp_mask, A.x[::-1], B.x[::-1], 1, thickness=1)
    ax.imshow(temp_mask, aspect='auto')
    fig.savefig(os.path.join(base_path, base_name + '_edge.png'), dpi=256)
    # region no need fig
    plt.close()


#########################################################################################
###################################### Class ############################################
#########################################################################################

class Element:
    def __init__(self, x, safe_count=0):
        assert type(x) is tuple
        assert type(x[0]) == int or type(x[0]) == Element
        assert type(x[1]) == int or type(x[1]) == Element
        self.x = x
        self.__score = None
        self.safe_count = safe_count

    def store_score(self, score):
        self.__score = score

    def get_score(self):
        return self.__score

    def equal(self, ele):
        if type(self.x[0]) != type(ele.x[0]):
            return False
        if type(self.x[0]) == int:
            # corner
            return True if self.x[0] == ele.x[0] and self.x[1] == ele.x[1] else False
        if type(self.x[0]) == Element:
            # edge
            if self.x[0].equal(ele.x[0]) and self.x[1].equal(ele.x[1]):
                return True
            if self.x[1].equal(ele.x[0]) and self.x[0].equal(ele.x[1]):
                return True
            return False
        raise BaseException('no implement type')


class regionCache():
    def __init__(self, datapath):
        self.cache = {}
        self.datapath = datapath

    def get_region(self, name):
        if name in self.cache.keys():
            return self.cache[name]
        gt_mask = np.load(os.path.join(self.datapath, name + '.npy'))
        if len(self.cache) == 5:
            self.cache.pop(list(self.cache.keys())[0])
            self.cache[name] = gt_mask
        return gt_mask


class imgCache():
    def __init__(self, datapath):
        self.cache = {}
        self.datapath = datapath

    def get_image(self, name):
        if name in self.cache.keys():
            return self.cache[name]
        img = skimage.img_as_float(plt.imread(os.path.join(self.datapath, 'rgb', name + '.jpg')))
        if len(self.cache) == 5:
            self.cache.pop(list(self.cache.keys())[0])
            self.cache[name] = img
        return img


class Graph:
    def __init__(self, corners, edges):
        corners, edges = sort_graph(corners, edges)

        self.__corners = []
        for corner_i in range(corners.shape[0]):
            self.__corners.append(
                Element(
                    tuple(
                        (int(corners[corner_i, 0]), int(corners[corner_i, 1]))
                    )
                )
            )
        self.__edges = []
        for edge_i in range(edges.shape[0]):
            self.__edges.append(Element((self.__corners[edges[edge_i, 0]], self.__corners[edges[edge_i, 1]])))
        self.__regions = []
        self.__regions.append(Element((0, 0)))  # we use entire region here

    @classmethod
    def initialFromTuple(cls, corners, edges):
        edge_index = []
        for item in edges:
            a = corners.index(item[0])
            b = corners.index(item[1])
            edge_index.append((a, b))
        edge_index = np.array(edge_index)
        corners = np.array(corners)
        return cls(corners, edge_index)

    def store_score(self, corner_score=None, edge_score=None, region_score=None):
        '''
        :param corner_score: np array size: len(corners)
        :param edge_score:  np array size: len(edges)
        :param region_score: np.array size: len(regions)
        :return:
        '''
        if corner_score is not None:
            for idx, element in enumerate(self.__corners):
                element.store_score(corner_score[idx])
        if edge_score is not None:
            for idx, element in enumerate(self.__edges):
                element.store_score(edge_score[idx])
        if region_score is not None:
            for idx, element in enumerate(self.__regions):
                element.store_score(region_score[idx])
        return

    def getCornersArray(self):
        c = []
        for ele in self.__corners:
            c.append(ele.x)
        return np.array(c)

    def getEdgesArray(self):
        c = []
        for ele in self.__edges:
            corner1 = ele.x[0]
            corner2 = ele.x[1]
            idx1 = self.__corners.index(corner1)
            idx2 = self.__corners.index(corner2)
            c.append([idx1, idx2])
        return np.array(c)

    def getCorners(self):
        return self.__corners

    def getRegions(self):
        return self.__regions

    def getEdges(self):
        return self.__edges

    def graph_score(self):
        corner_score = 0
        for ele in self.__corners:
            corner_score += ele.get_score()
        edge_score = 0
        for ele in self.__edges:
            edge_score += ele.get_score()
        region_score = 0
        for ele in self.__regions:
            region_score += ele.get_score()
        return score_weights[0] * corner_score + score_weights[1] * edge_score + score_weights[2] * region_score

    def corner_score(self):
        corner_score = 0
        for ele in self.__corners:
            corner_score += ele.get_score()
        return corner_score

    def edge_score(self):
        edge_score = 0
        for ele in self.__edges:
            edge_score += ele.get_score()
        return edge_score

    def region_score(self):
        region_score = 0
        for ele in self.__regions:
            region_score += ele.get_score()
        return region_score

    def remove(self, ele):
        '''
        :param ele: remove eles as well as some other related elements
        :return: set() of removed elements
        '''
        # corner
        removed = set()
        if ele in self.__corners:
            self.__corners.remove(ele)
            removed.add(ele)
            # remove edge that has the corner
            for idx in reversed(range(len(self.__edges))):
                edge_ele = self.__edges[idx]
                if ele in edge_ele.x:
                    removed = removed.union(self.remove(edge_ele))
        # edge
        elif ele in self.__edges:
            self.__edges.remove(ele)
            removed.add(ele)
            corner1 = ele.x[0]
            corner2 = ele.x[1]
            if corner1.safe_count == 0:
                # can be delete
                _count = 0
                for edge_ele in self.__edges:
                    if corner1 in edge_ele.x:
                        _count += 1
                if _count == 0:
                    removed = removed.union(self.remove(corner1))
            if corner2.safe_count == 0:
                # can be delete
                _count = 0
                for edge_ele in self.__edges:
                    if corner2 in edge_ele.x:
                        _count += 1
                if _count == 0:
                    removed = removed.union(self.remove(corner2))
        return removed

    def has_edge(self, ele1, ele2):
        """
        :param ele1: corner1
        :param ele2: corner2
        :return: edge or none
        """
        for edge_ele in self.__edges:
            if ele1 in edge_ele.x and ele2 in edge_ele.x:
                return edge_ele
        return None

    def add_edge(self, ele1, ele2):
        temp = self.has_edge(ele1, ele2)
        if temp is not None:
            temp.safe_count = SAFE_NUM
            return temp
        new_ele = Element((ele1, ele2), safe_count=SAFE_NUM)
        self.__edges.append(new_ele)
        return new_ele

    def add_corner(self, ele):
        for corner in self.__corners:
            if corner.x == ele.x:
                corner.safe_count = SAFE_NUM
                return corner
        ele.safe_count = SAFE_NUM
        self.__corners.append(ele)
        return ele

    def add_corner_v2(self, ele):
        # if new corner is near a existed corner, return the existed corner
        # if new corner is on an edge, split edge
        for corner in self.__corners:
            if l2_distance(corner.x, ele.x) < 5:
                corner.safe_count = SAFE_NUM
                return corner
        min_d = 256
        the_edge = None
        for edge in self.__edges:
            temp = get_distance_of_corner_and_edge(edge.x[0].x, edge.x[1].x, ele.x)
            if temp < min_d:
                min_d = temp
                the_edge = edge
        if min_d < 3:
            # split edge
            corner1 = the_edge.x[0]
            corner2 = the_edge.x[1]
            new_ele = Element((corner1, ele), safe_count=the_edge.safe_count)
            self.__edges.append(new_ele)
            new_ele = Element((corner2, ele), safe_count=the_edge.safe_count)
            self.__edges.append(new_ele)
            self.__edges.remove(the_edge)
        ele.safe_count = SAFE_NUM
        self.__corners.append(ele)
        return ele

    def checkColinearCorner(self, ele):
        if self.getCornerDegree(ele) != 2:
            return False
        edge_in = []
        for edge_ele in self.__edges:
            if ele in edge_ele.x:
                edge_in.append(edge_ele)
                if len(edge_in) == 2:
                    break
        two_neighbor = {edge_in[0].x[0], edge_in[0].x[1], edge_in[1].x[0], edge_in[1].x[1]}
        two_neighbor.remove(ele)
        two_neighbor = tuple(two_neighbor)
        if self.has_edge(two_neighbor[0], two_neighbor[1]) is not None:
            return False

        line1 = np.array(ele.x) - np.array(two_neighbor[0].x)
        line2 = np.array(two_neighbor[1].x) - np.array(ele.x)
        cos = np.dot(line1, line2) / (np.linalg.norm(line1) * np.linalg.norm(line2))
        cos = min(1, max(-1, cos))
        if np.arccos(cos) < np.pi / 9:  # 20 degree
            return True
        return False

    def checkIntersectionEdge(self, ele=None):
        if ele is None:
            for edge_i in range(len(self.__edges)):
                for edge_j in range(edge_i + 1, len(self.__edges)):
                    if check_intersection(self.__edges[edge_i], self.__edges[edge_j]):
                        return True
            return False
        for edge_ele in self.__edges:
            if ele == edge_ele:
                continue
            if check_intersection(edge_ele, ele):
                return True
        return False

    def getCornerDegree(self, ele):
        degree = 0
        for edge_ele in self.__edges:
            if ele in edge_ele.x:
                degree += 1
        return degree

    def getEdgeConnected(self, ele):
        out_ = set()
        if type(ele.x[0]) == int:
            # corner
            for edge_ele in self.__edges:
                if ele in edge_ele.x:
                    out_.add(edge_ele)
            return out_
        if type(ele.x[0]) == Element:
            # Edge
            out_ = out_.union(self.getEdgeConnected(ele.x[0]))
            out_ = out_.union(self.getEdgeConnected(ele.x[1]))
            if ele in out_:
                out_.remove(ele)
            return out_

    def getNeighborCorner(self, ele):
        out_ = set()
        for edge_ele in self.__edges:
            if ele == edge_ele.x[0]:
                out_.add(edge_ele.x[1])
            if ele == edge_ele.x[1]:
                out_.add(edge_ele.x[0])
        return out_

    def getRealElement(self, ele):
        # edge
        if type(ele.x[0]) == Element:
            for e in self.__edges:
                if (e.x[0].x == ele.x[0].x and e.x[1].x == ele.x[1].x) or \
                        (e.x[1].x == ele.x[0].x and e.x[0].x == ele.x[1].x):
                    return e
            raise BaseException("no same edge exists.")
        # corner
        elif type(ele.x[0]) == int:
            for c in self.__corners:
                if c.x == ele.x:
                    return c
            raise BaseException("no same corner exists.")

    def copy(self):
        corners = self.getCornersArray()
        edges = self.getEdgesArray()
        new_graph = Graph(corners, edges)
        for idx, ele in enumerate(self.__corners):
            new_graph.__corners[idx].store_score(self.__corners[idx].get_score())
        for idx, ele in enumerate(self.__edges):
            new_graph.__edges[idx].store_score(self.__edges[idx].get_score())
        for idx, ele in enumerate(self.__regions):
            new_graph.__regions[idx].store_score(self.__regions[idx].get_score)
        return new_graph

    def update_safe_count(self):
        for ele in self.__corners:
            if ele.safe_count > 0:
                ele.safe_count -= 1
        for ele in self.__edges:
            if ele.safe_count > 0:
                ele.safe_count -= 1

    def isNeighbor(self, element1, element2):
        '''
        :param element1:
        :param element2:
        :return: True / False
        '''
        if element1 == element2:
            return False
        if type(element1.x[0]) != type(element2.x[0]):
            # corner and edge
            return False
        if type(element1.x[0]) == int:
            # both are corner type
            for edge_ele in self.__edges:
                if edge_ele.x[0] == element1 and edge_ele.x[1] == element2:
                    return True
                if edge_ele.x[0] == element2 and edge_ele.x[1] == element1:
                    return True
            return False
        if type(element1.x[0]) == Element:
            # both are edge type
            if len({element1.x[0], element1.x[1], element2.x[0], element2.x[1]}) < 4:
                return True
            return False

    def equal(self, graph):
        if len(self.__corners) != len(graph.__corners) or \
                len(self.__edges) != len(graph.__edges):
            return False
        for corner_i in range(len(self.__corners)):
            if self.__corners[corner_i].equal(graph.__corners[corner_i]) is False:
                return False
        for edge_i in range(len(self.__edges)):
            if self.__edges[edge_i].equal(graph.__edges[edge_i]) is False:
                return False

        return True


class Candidate:
    def __init__(self, graph, name, corner_existed_before, edge_existed_before):
        '''
        :param graph: Class graph
        :param name: string, data name
        :param corner_existed_before: dict {(x_i,y_i):c_1 ...} indicates counts for corresponding corners, after one search,
                                     counts -= 1, if count == 0, remove from the set.
        :param edge_existed_before: dict {((x_i1,y_i1),(x_i2,y_i2)):ci}
        '''
        self.graph = graph
        self.name = name
        self.corner_existed_before = corner_existed_before
        self.edge_existed_before = edge_existed_before

    @classmethod
    def initial(cls, graph, name):
        return cls(graph, name, {}, {})

    def update(self):
        # all the existed before elements count - 1
        for key in self.corner_existed_before.keys():
            self.corner_existed_before[key] -= 1
        for key in self.edge_existed_before.keys():
            self.edge_existed_before[key] -= 1

        # check if some need to remove from existed before set
        for key in list(self.corner_existed_before.keys()):
            if self.corner_existed_before[key] == 0:
                self.corner_existed_before.pop(key)

        for key in list(self.edge_existed_before.keys()):
            if self.edge_existed_before[key] == 0:
                self.edge_existed_before.pop(key)

        # update graph
        self.graph.update_safe_count()

    def copy(self):
        corner_existed_before = self.corner_existed_before.copy()
        edge_existed_before = self.edge_existed_before.copy()
        new_graph = self.graph.copy()
        return Candidate(new_graph, self.name, corner_existed_before, edge_existed_before)

    def removable(self, ele):
        '''
        :param x: input is element
        :return:
        '''
        assert type(ele) == Element
        # edge
        return True if ele.safe_count == 0 else False

    def addable(self, ele):
        if type(ele) == Element:
            if type(ele.x[0]) == Element:
                # edge
                for edge in self.graph.getEdges():
                    c1 = edge.x[0]
                    c2 = edge.x[1]
                    if (ele.x[0].x == c1.x and ele.x[1].x == c2.x) or \
                            (ele.x[1].x == c1.x and ele.x[0].x == c2.x):
                        # already existed
                        return False
                corner1_loc = ele.x[0].x
                corner2_loc = ele.x[1].x
                if (corner1_loc, corner2_loc) in self.edge_existed_before.keys() or \
                        (corner2_loc, corner1_loc) in self.edge_existed_before.keys():
                    return False
                return True
            else:
                # corner
                for corner in self.graph.getCorners():
                    if l2_distance(ele.x, corner.x) < TWO_CORNER_MINIMUM_DISTANCE:
                        # already existed
                        return False
                if ele.x in self.corner_existed_before.keys():
                    return False
                return True
        else:  # (x,y) or ((x1,y1),(x2,y2))
            if type(ele[0]) == tuple:
                # edge
                corner1_loc = ele[0]
                corner2_loc = ele[1]
                for edge in self.graph.getEdges():
                    c1 = edge.x[0]
                    c2 = edge.x[1]
                    if (corner1_loc == c1.x and corner2_loc == c2.x) or \
                            (corner2_loc == c1.x and corner1_loc == c2.x):
                        # already existed
                        return False
                if (corner1_loc, corner2_loc) in self.edge_existed_before.keys() or \
                        (corner2_loc, corner1_loc) in self.edge_existed_before.keys():
                    return False
                return True
            else:
                # corner
                for corner in self.graph.getCorners():
                    if l2_distance(ele, corner.x) < TWO_CORNER_MINIMUM_DISTANCE:
                        # already existed
                        return False
                if ele in self.corner_existed_before.keys():
                    return False
                return True

    def addCorner(self, ele):
        if ele.x in self.corner_existed_before.keys():
            raise BaseException('cannot add the corner')
        new_ele = self.graph.add_corner(ele)  # possible changed
        return new_ele

    def addCorner_v2(self, ele):
        if ele.x in self.corner_existed_before.keys():
            raise BaseException('cannot add the corner')
        new_ele = self.graph.add_corner_v2(ele)
        return new_ele

    def addEdge(self, ele1, ele2):
        corner1 = ele1
        corner2 = ele2
        assert corner1 in self.graph.getCorners()
        assert corner2 in self.graph.getCorners()
        if (corner1.x, corner2.x) in self.edge_existed_before.keys() or \
                (corner2.x, corner1.x) in self.edge_existed_before.keys():
            raise BaseException('cannot add the edge')
        new_ele = self.graph.add_edge(corner1, corner2)
        return new_ele

    def removeCorner(self, ele):
        if ele.x in self.corner_existed_before.keys():
            raise BaseException('already existed.')
        self.corner_existed_before[ele.x] = SAFE_NUM

    def removeEdge(self, ele):
        corner1 = ele.x[0]
        corner2 = ele.x[1]
        loc1 = corner1.x
        loc2 = corner2.x
        if (loc1[0] > loc2[0]) or (loc1[0] == loc2[0] and loc1[1] > loc2[1]):
            loc1 = corner2.x
            loc2 = corner1.x
        if (loc1, loc2) in self.edge_existed_before.keys():
            raise BaseException('already existed.')
        self.edge_existed_before[(loc1, loc2)] = SAFE_NUM

    def generate_new_candidate_remove_a_colinear_corner(self, ele):
        # need to check if ele is a colinear corner before
        new_candidate = self.copy()
        new_graph = new_candidate.graph
        ele = new_graph.getRealElement(ele)

        # find two neighbor corners
        temp = set()
        for element in new_graph.getEdgeConnected(ele):
            # edge
            if type(element.x[0]) == Element:
                temp.add(element.x[0])
                temp.add(element.x[1])
        temp.remove(ele)
        temp = tuple(temp)
        assert len(temp) == 2

        # add edge to two neighbor corners
        # (add before remove, in case the neighbor corners will be removed by zero degree)
        # special case no need to check existed_before, instead remove if in existed_before dict
        added = new_graph.add_edge(temp[0], temp[1])
        if (temp[0].x, temp[1].x) in self.edge_existed_before.keys():
            self.edge_existed_before.pop((temp[0].x, temp[1].x))
        if (temp[1].x, temp[0].x) in self.edge_existed_before.keys():
            self.edge_existed_before.pop((temp[1].x, temp[0].x))

        # remove
        removed = new_graph.remove(ele)

        # add removed elements into existed before
        for element in removed:
            # edge
            if type(element.x[0]) == Element:
                new_candidate.removeEdge(element)
            # corner
            elif type(element.x[0]) == int:
                new_candidate.removeCorner(element)
            else:
                raise BaseException('wrong type.')

        # modify scores that need to be recounted
        # all corners are recounted
        for element in new_graph.getCorners():
            element.store_score(None)

        # edges that are neighbors to the removed edges OR new edges will be recounted
        for element in new_graph.getEdges():
            for modified_ele in removed.union({added}):
                if new_graph.isNeighbor(element, modified_ele):
                    element.store_score(None)
                    break

        # all regions are recounted
        for element in new_graph.getRegions():
            element.store_score(None)

        return new_candidate

    def generate_new_candidate_remove_a_corner(self, ele):
        # need to check if ele is removable before call this method
        new_candidate = self.copy()
        new_graph = new_candidate.graph
        ele = new_graph.getRealElement(ele)
        removed = new_graph.remove(ele)

        # add removed elements into existed before
        for element in removed:
            # edge
            if type(element.x[0]) == Element:
                corner1 = element.x[0]
                corner2 = element.x[1]
                loc1 = corner1.x
                loc2 = corner2.x
                if (loc1[0] > loc2[0]) or (loc1[0] == loc2[0] and loc1[1] > loc2[1]):
                    loc1 = corner2.x
                    loc2 = corner1.x
                if (loc1, loc2) in self.edge_existed_before.keys():
                    raise BaseException('already existed.')
                new_candidate.edge_existed_before[(loc1, loc2)] = SAFE_NUM
            # corner
            elif type(element.x[0]) == int:
                if element.x in self.corner_existed_before.keys():
                    raise BaseException('already existed.')
                new_candidate.corner_existed_before[element.x] = SAFE_NUM
            else:
                raise BaseException('wrong type.')

        # modify scores that need to be recounted
        # all corners are recounted
        for element in new_graph.getCorners():
            element.store_score(None)

        # edges that are neighbors to the removed edges will be recounted
        for element in new_graph.getEdges():
            for removed_ele in removed:
                if new_graph.isNeighbor(element, removed_ele):
                    element.store_score(None)
                    break

        # all regions are recounted
        for element in new_graph.getRegions():
            element.store_score(None)

        return new_candidate

    def generate_new_candidate_add_an_edge(self, ele1, ele2):
        # need to check addable before call this method
        new_candidate = self.copy()
        new_graph = new_candidate.graph
        ele1 = new_graph.getRealElement(ele1)
        ele2 = new_graph.getRealElement(ele2)

        # add edge
        new_ele = new_candidate.addEdge(ele1, ele2)

        # modify scores that need to be recounted
        # all corners are recounted
        for element in new_graph.getCorners():
            element.store_score(None)

        # edges that are neighbors to the added edges will be recounted
        for element in new_graph.getEdges():
            if new_graph.isNeighbor(element, new_ele):
                element.store_score(None)

        # all regions are recounted
        for element in new_graph.getRegions():
            element.store_score(None)

        return new_candidate

    def generate_new_candidate_remove_an_edge(self, ele):
        # need to check if ele is removable before call this method
        new_candidate = self.copy()
        new_graph = new_candidate.graph
        ele = new_graph.getRealElement(ele)
        removed = new_graph.remove(ele)

        # add removed elements into existed before
        for element in removed:
            # edge
            if type(element.x[0]) == Element:
                corner1 = element.x[0]
                corner2 = element.x[1]
                loc1 = corner1.x
                loc2 = corner2.x
                if (loc1[0] > loc2[0]) or (loc1[0] == loc2[0] and loc1[1] > loc2[1]):
                    loc1 = corner2.x
                    loc2 = corner1.x
                if (loc1, loc2) in self.edge_existed_before.keys():
                    raise BaseException('already existed.')
                new_candidate.edge_existed_before[(loc1, loc2)] = SAFE_NUM
            # corner
            elif type(element.x[0]) == int:
                if element.x in self.corner_existed_before.keys():
                    raise BaseException('already existed.')
                new_candidate.corner_existed_before[element.x] = SAFE_NUM
            else:
                raise BaseException('wrong type.')

        # modify scores that need to be recounted
        # all corners are recounted
        for element in new_graph.getCorners():
            element.store_score(None)

        # edges that are neighbors to the removed edges will be recounted
        for element in new_graph.getEdges():
            for removed_ele in removed:
                if new_graph.isNeighbor(element, removed_ele):
                    element.store_score(None)
                    break

        # all regions are recounted
        for element in new_graph.getRegions():
            element.store_score(None)

        return new_candidate

    def generate_new_candidate_add_a_new_triangle(self, ele_new, ele1, ele2):
        # this method is to add a new corner as well as two new edges into the graph
        # need to check addable of ele_new before call this method
        new_candidate = self.copy()
        new_graph = new_candidate.graph
        ele1 = new_graph.getRealElement(ele1)
        ele2 = new_graph.getRealElement(ele2)

        # add corner
        ele_new = new_candidate.addCorner(ele_new)  # ele_new possible change

        # no score need to be recounted in current situation

        # add two_new edge (ele1, ele_new) and (ele2, ele_new)
        new_candidate = new_candidate.generate_new_candidate_add_an_edge(ele_new, ele1)
        new_candidate = new_candidate.generate_new_candidate_add_an_edge(ele_new, ele2)

        return new_candidate

    def generate_new_candidate_add_a_corner(self, ele):
        # need to check addable of ele before call this method
        new_candidate = self.copy()
        new_graph = new_candidate.graph

        # add corner
        ele = new_candidate.addCorner(ele)

        # modify scores that need to be recounted
        # all corners are recounted
        for element in new_graph.getCorners():
            element.store_score(None)

        # no edge need to be recounted
        # all regions are recounted
        for element in new_graph.getRegions():
            element.store_score(None)

        return new_candidate

    def equal(self, candidate):
        return self.graph.equal(candidate.graph)
