import os
import json
import cv2
import numpy as np
from collections import defaultdict
from scipy import ndimage


def generate_graph(annot, image_path, out_path):
    lines = annot['lines']
    junctions = annot['junctions']
    line_junc_mat = np.array(annot['lineJunctionMatrix'])
    planes = annot['planes']
    plane_line_mat = annot['planeLineMatrix']
    plane_to_line = np.array(annot['planeLineMatrix'])
    line_to_plane = plane_to_line.T
    semantics = annot['semantics']

    all_room_edges = get_room_edges(semantics, planes, lines, junctions, plane_to_line, line_junc_mat)

    all_room_edges = filter_rooms(all_room_edges, im_size=256)

    all_colinear_pairs = find_all_colinear_paris(all_room_edges)
    colinear_sets = combine_colinear_edges(all_colinear_pairs)

    for colinear_set in colinear_sets:
        edges_to_merge = list(colinear_set)
        edges_to_merge = sorted(edges_to_merge, key=lambda x: -x[0])
        merged_edges = merge_edges(edges_to_merge)

        for merged_edge, old_edge in zip(merged_edges, edges_to_merge):
            if len(merged_edge) > 0:
                assert merged_edge[0][0] == old_edge[0]
                room_idx = merged_edge[0][0]

                if len(merged_edge) == 1 and merged_edge[0] == old_edge:
                    continue
                # change room graph accordingly
                replaced_idx = all_room_edges[room_idx].index(old_edge[1])
                all_room_edges[room_idx].pop(replaced_idx)
                for new_idx, new_edge in enumerate(merged_edge):
                    insert_idx = new_idx + replaced_idx
                    all_room_edges[room_idx].insert(insert_idx, new_edge[1])
            else:
                room_idx = old_edge[0]
                replaced_idx = all_room_edges[room_idx].index(old_edge[1])
                all_room_edges[room_idx].pop(replaced_idx)

    # take intersection for every rooms to recover the room structure
    refined_room_edges = [adjust_room_edges(room_edges) for room_edges in all_room_edges]

    # clean every room loop by removing I-shape corners
    cleaned_room_edges = clean_room_edges(refined_room_edges)

    global_graph = defaultdict(list)
    for room_edges in cleaned_room_edges:
        for edge in room_edges:
            c1, c2 = edge
            global_graph[c1] += [c2, ]
            global_graph[c2] += [c1, ]
    for corner in global_graph:
        global_graph[corner] = list(set(global_graph[corner]))

    annot_path = os.path.join(out_path, 'annot.npy')
    np.save(annot_path, global_graph)

    # draw the planar graph on the density map
    viz_image = cv2.imread(image_path)
    for c, connections in global_graph.items():
        for other_c in connections:
            cv2.line(viz_image, (int(c[0]), int(c[1])), (int(other_c[0]), int(other_c[1])), (255, 255, 0), 2)
    for c in global_graph.keys():
        cv2.circle(viz_image, (int(c[0]), int(c[1])), 3, (0, 0, 255), -1)

    # viz_image = np.zeros([256, 266, 3]).astype(np.uint8)
    # room_idx = 0
    # for room_edges in cleaned_room_edges:
    #     for line_idx, edge in enumerate(room_edges):
    #         c1, c2 = np.array(edge).astype(np.int)
    #         cv2.line(viz_image, tuple(c1), tuple(c2), (255, 255, 0), 2)
    #         cv2.circle(viz_image, tuple(c1), 3, (0, 0, 255), -1)
    #         cv2.circle(viz_image, tuple(c2), 3, (0, 0, 255), -1)
    #     room_idx += 1
    cv2.imwrite(os.path.join(out_path, 'planar_graph.png'), viz_image)


def get_room_edges(semantics, planes, lines, junctions, plane_to_line, line_junc_mat):
    room_edges = list()
    for semantic in semantics:
        plane_ids = semantic['planeID']
        label = semantic['type']
        if label in ['door', 'window', 'outwall']:  # skip non-room elements
            continue
        all_planes = [planes[idx] for idx in plane_ids]
        floor_planes = [plane for plane in all_planes if plane['type'] == 'floor']
        assert len(floor_planes) == 1, 'There should be only one floor for each room'
        floor_plane = floor_planes[0]
        floor_plane_id = floor_plane['ID']
        line_ids = np.where(plane_to_line[floor_plane_id])[0].tolist()
        floor_lines_raw = [lines[line_id] for line_id in line_ids]
        floor_lines = list()
        for line_idx, floor_line in enumerate(floor_lines_raw):
            c_id_1, c_id_2 = np.where(line_junc_mat[floor_line['ID']])[0].tolist()
            c1 = tuple(junctions[c_id_1]['coordinate'][:2])
            c2 = tuple(np.array(junctions[c_id_2]['coordinate'][:2]))
            if c1 == c2:
                continue
            floor_lines.append((c1, c2))

        floor_lines = list(set(floor_lines))  # remove duplications
        floor_lines = sort_room_edges(floor_lines)
        room_edges.append(floor_lines)
    return room_edges


def sort_room_edges(lines):
    cur_id = 0
    picked = [False] * len(lines)
    id_list = [0, ]
    while len(id_list) < len(lines):
        line = lines[cur_id]
        picked[cur_id] = True
        check_ = [(line[1] in other) and not picked[other_idx] for other_idx, other in enumerate(lines)]
        next_ids = np.nonzero(check_)[0]
        try:
            assert len(next_ids) == 1
        except:
            raise WrongRoomError('Invalid room shape')
        next_id = next_ids[0]
        id_list.append(next_id)
        if lines[next_id][1] == line[1]:  # swap the two endpoints, to make the loop valid.
            lines[next_id] = (lines[next_id][1], lines[next_id][0])
        cur_id = next_id
        if lines[next_id][1] == lines[0][0]:  # already form a closed loop, then skip the remaining lines
            break
    sorted_lines = [lines[idx] for idx in id_list]
    return sorted_lines


def find_all_colinear_paris(all_room_edges):
    colinear_pairs = list()
    for room_idx, room_edges in enumerate(all_room_edges):
        for edge_idx, edge in enumerate(room_edges):
            for other_room_idx, other_edges in enumerate(all_room_edges):
                if other_room_idx < room_idx:
                    continue
                for other_edge_idx, other_edge in enumerate(other_edges):
                    if other_room_idx == room_idx and other_edge_idx <= edge_idx:
                        continue
                    if _check_colinear(edge, other_edge, line_dist_th=8):
                        ele1 = (room_idx, edge)
                        ele2 = (other_room_idx, other_edge)
                        colinear_pairs.append([ele1, ele2])
    return colinear_pairs


def combine_colinear_edges(colinear_pairs):
    all_colinear_sets = list()
    all_pairs = list(colinear_pairs)  # make a copy of the input list
    combined = [False] * len(colinear_pairs)

    while len(all_pairs) > 0:
        colinear_set = _combine_colinear_pairs(0, all_pairs, combined)
        all_colinear_sets.append(colinear_set)
        all_pairs = [all_pairs[i] for i in range(len(all_pairs)) if combined[i] is False]
        combined = [False] * len(all_pairs)
    return all_colinear_sets


def _combine_colinear_pairs(idx, all_pairs, combined):
    colinear_set = set(all_pairs[idx])
    combined[idx] = True
    for other_idx, pair in enumerate(all_pairs):
        if not combined[other_idx] and (
                all_pairs[idx][0] in all_pairs[other_idx] or all_pairs[idx][1] in all_pairs[other_idx]):
            colinear_set = colinear_set.union(_combine_colinear_pairs(other_idx, all_pairs, combined))
    return colinear_set


def _check_colinear(e1, e2, line_dist_th=8):
    # first check whether two line segments are parallel to each other, if not, return False directly
    len_e1 = len_edge(e1)
    len_e2 = len_edge(e2)
    # we need to always make e2 the shorter one
    if len_e1 < len_e2:
        e1, e2 = e2, e1
    v1_01 = (e1[1][0] - e1[0][0], e1[1][1] - e1[0][1])
    v1_10 = (e1[0][0] - e1[1][0], e1[0][1] - e1[1][1])
    v2_01 = (e2[1][0] - e2[0][0], e2[1][1] - e2[0][1])
    v2_10 = (e2[0][0] - e2[1][0], e2[0][1] - e2[1][1])
    len_1 = np.sqrt(v1_01[0] ** 2 + v1_01[1] ** 2)
    len_2 = np.sqrt(v2_01[0] ** 2 + v2_01[1] ** 2)
    if len_1 == 0 or len_2 == 0:
        cos = 0
    else:
        cos = (v1_01[0] * v2_01[0] + v1_01[1] * v2_01[1]) / (len_1 * len_2)
    if abs(cos) > 0.99:
        # then check the distance between two parallel lines
        len_10_20 = len_edge((e1[0], e2[0]))
        len_10_21 = len_edge((e1[0], e2[1]))
        len_11_20 = len_edge((e1[1], e2[0]))
        len_11_21 = len_edge((e1[1], e2[1]))

        # two endpoints are very close, then we can say these two edges are colinear
        if np.min([len_10_20, len_10_21, len_11_20, len_11_21]) <= 5:
            return True
        # otherwise we need to check the distance first
        v_10_20 = (e2[0][0] - e1[0][0], e2[0][1] - e1[0][1])
        cos_11_10_20 = (v1_01[0] * v_10_20[0] + v1_01[1] * v_10_20[1]) / (len_1 * len_10_20)
        sin_11_10_20 = np.sqrt(1 - cos_11_10_20 ** 2)
        dist_20_e1 = len_10_20 * sin_11_10_20
        if dist_20_e1 <= line_dist_th:
            # we need two check whether they have some overlaps
            v_11_20 = (e2[0][0] - e1[1][0], e2[0][1] - e1[1][1])
            cos_10_11_20 = (v1_10[0] * v_11_20[0] + v1_10[1] * v_11_20[1]) / (len_1 * len_11_20)
            if cos_11_10_20 >= 0 and cos_10_11_20 >= 0:
                return True
            v_10_21 = (e2[1][0] - e1[0][0], e2[1][1] - e1[0][1])
            cos_11_10_21 = (v1_01[0] * v_10_21[0] + v1_01[1] * v_10_21[1]) / (len_1 * len_10_21)
            v_11_21 = (e2[1][0] - e1[1][0], e2[1][1] - e1[1][1])
            cos_10_11_21 = (v1_10[0] * v_11_21[0] + v1_10[1] * v_11_21[1]) / (len_1 * len_11_21)
            if cos_11_10_21 >= 0 and cos_10_11_21 >= 0:
                return True
            return False
        else:
            # if the two line segments have distance > 3, we can say they are not colinear
            return False
    else:
        return False


def merge_edges(edges):
    base_e = edges[0][1]
    merged_edges = [edges[0], ]
    base_len = np.sqrt((base_e[1][0] - base_e[0][0]) ** 2 + (base_e[1][1] - base_e[0][1]) ** 2)
    base_unit_v = ((base_e[1][0] - base_e[0][0]) / base_len, (base_e[1][1] - base_e[0][1]) / base_len)

    for edge in edges[1:]:
        room_idx = edge[0]
        e = edge[1]
        v_b0e0 = (e[0][0] - base_e[0][0], e[0][1] - base_e[0][1])
        proj_len = (v_b0e0[0] * base_unit_v[0] + v_b0e0[1] * base_unit_v[1])
        proj_e0 = (int(base_e[0][0] + base_unit_v[0] * proj_len), int(base_e[0][1] + base_unit_v[1] * proj_len))
        proj_e1 = (int(proj_e0[0] + e[1][0] - e[0][0]), int(proj_e0[1] + e[1][1] - e[0][1]))
        new_e = (proj_e0, proj_e1)
        new_edge = (room_idx, new_e)
        merged_edges.append(new_edge)

    adjusted_merged_edges = adjust_colinear_edges(merged_edges)

    return adjusted_merged_edges


def adjust_colinear_edges(edges):
    base_corner = (edges[0][0], edges[0][1][0])
    all_corners = [base_corner, (edges[0][0], edges[0][1][1])]
    for edge in edges[1:]:
        all_corners.append((edge[0], edge[1][0]))
        all_corners.append((edge[0], edge[1][1]))
    unit_v = unit_v_edge(edges[0][1])
    corner_projs = list()
    # FIXME: need to fix the corner coords here, it's wrong now!! they are not merged...
    for room, other_c in all_corners:
        v_base_c = (other_c[0] - base_corner[1][0], other_c[1] - base_corner[1][1])
        proj = (unit_v[0] * v_base_c[0] + unit_v[1] * v_base_c[1])
        corner_projs.append(proj)
    order = np.argsort(corner_projs).tolist()

    # # merge corners that are too close to the prev corner
    # for o_idx, corner_idx in enumerate(order[1:]):
    #     corner = all_corners[corner_idx][1]
    #     prev_idx = order[o_idx]
    #     prev_corner = all_corners[prev_idx][1]
    #     dist = len_edge((corner, prev_corner))
    #     if dist <= 5:
    #         all_corners[corner_idx] = (all_corners[corner_idx][0], prev_corner)

    adjusted_edges = list()
    for idx, edge in enumerate(edges):
        room_idx = edge[0]
        idx_1 = idx * 2
        idx_2 = idx * 2 + 1
        adj_idx_1 = order.index(idx_1)
        adj_idx_2 = order.index(idx_2)
        step_direction = 1 if adj_idx_2 > adj_idx_1 else -1
        adjusted_edge = list()
        for o_idx in range(adj_idx_1, adj_idx_2, step_direction):
            c_idx = order[o_idx]
            next_c_idx = order[o_idx + step_direction]
            segment = (room_idx, (all_corners[c_idx][1], all_corners[next_c_idx][1]))
            if len_edge(segment[1]) == 0:
                continue
            adjusted_edge.append(segment)
        adjusted_edges.append(adjusted_edge)
    return adjusted_edges


def adjust_room_edges(room_edges):
    refined_room_edges = list()

    init_room_edges = list(room_edges)
    for edge_i, edge in enumerate(room_edges):
        next_i = edge_i
        while True:
            next_i = next_i + 1 if next_i < len(room_edges) - 1 else 0
            next_edge = room_edges[next_i]
            if next_edge[0] != next_edge[1]:
                break
        if edge[1] == next_edge[0]:  # no need for refining
            refined_room_edges.append(edge)
        else:  # the two corners disagree, refinement is required
            if edge[0] == edge[1]:
                print('skip collasped edge')
                continue
            unit_edge = unit_v_edge(edge)
            ext_edge = ((edge[0][0] - unit_edge[0] * 50, edge[0][1] - unit_edge[1] * 50),
                        (edge[1][0] + unit_edge[0] * 50, edge[1][1] + unit_edge[1] * 50))
            unit_next = unit_v_edge(next_edge)
            ext_next = ((next_edge[0][0] - unit_next[0] * 50, next_edge[0][1] - unit_next[1] * 50),
                        (next_edge[1][0] + unit_next[0] * 50, next_edge[1][1] + unit_next[1] * 50))
            intersec = get_intersection(ext_edge[0], ext_edge[1], ext_next[0], ext_next[1])
            try:
                assert intersec is not None
            except:
                print('no intersect, move endpoint directly')
                intersec = next_edge[0]
            intersec = (int(np.round(intersec[0])), int(np.round(intersec[1])))
            refined_edge = (edge[0], intersec)
            refined_room_edges.append(refined_edge)
            room_edges[edge_i] = refined_edge
            room_edges[next_i] = (intersec, next_edge[1])
            if next_i < edge_i:
                refined_room_edges[next_i] = room_edges[next_i]

    # drop collapsed edges
    refined_room_edges = [edge for edge in refined_room_edges if edge[0] != edge[1]]
    for edge_i in range(len(refined_room_edges)):
        next_i = edge_i + 1 if edge_i < len(refined_room_edges) - 1 else 0
        if refined_room_edges[edge_i][1] != refined_room_edges[next_i][0]:
            new_edge = (refined_room_edges[edge_i][0], refined_room_edges[next_i][0])
            refined_room_edges[edge_i] = new_edge
    return refined_room_edges


def clean_room_edges(all_room_edges):
    refined_room_paths = [_extract_room_path(room_edges) for room_edges in all_room_edges]
    corner_to_room = defaultdict(list)
    for room_idx, room_path in enumerate(refined_room_paths):
        for corner in room_path:
            corner_to_room[corner].append(room_idx)
    # remove I-shape corner used by only one room
    for room_idx, room_edges in enumerate(all_room_edges):
        cp_room_edges = list(room_edges)
        rm_flag = True
        while rm_flag:
            rm_flag = False
            for edge_i, edge in enumerate(cp_room_edges):
                prev_i = edge_i - 1
                prev_edge = cp_room_edges[prev_i]
                if _check_colinear(prev_edge, edge, line_dist_th=5):
                    rm_candidate = edge[0]
                    if len(corner_to_room[rm_candidate]) == 1 and corner_to_room[rm_candidate][0] == room_idx:
                        cp_room_edges[prev_i] = (prev_edge[0], edge[1])
                        rm_flag = True
                        cp_room_edges.pop(edge_i)
                        break
                next_i = edge_i + 1 if edge_i < len(cp_room_edges) - 1 else 0
                next_edge = cp_room_edges[next_i]
                if _check_colinear(next_edge, edge, line_dist_th=5):
                    rm_candidate = edge[1]
                    if len(corner_to_room[rm_candidate]) == 1 and corner_to_room[rm_candidate][0] == room_idx:
                        cp_room_edges[next_i] = (edge[0], next_edge[1])
                        rm_flag = True
                        cp_room_edges.pop(edge_i)
                        break
        if len(cp_room_edges) != len(room_edges):
            all_room_edges[room_idx] = cp_room_edges

    corner_to_room = get_corner_to_room(all_room_edges)
    all_corners = list(corner_to_room.keys())
    corners_to_merge = find_corners_to_merge(all_corners, corner_to_room)
    while corners_to_merge is not None:
        num_aff = [len(corner_to_room[x]) for x in corners_to_merge]
        order = np.argsort(num_aff)[::-1]
        base_corner = corners_to_merge[order[0]]
        for corner in corners_to_merge:
            if corner == base_corner:
                continue
            all_room_edges = move_corner(corner, base_corner, corner_to_room, all_room_edges)

        corner_to_room = get_corner_to_room(all_room_edges)
        all_corners = list(corner_to_room.keys())
        corners_to_merge = find_corners_to_merge(all_corners, corner_to_room)

    # for room_idx, room_edges in enumerate(all_room_edges):
    #     cp_room_edges = list(room_edges)
    #     rm_flag = True
    #     while rm_flag:
    #         rm_flag = False
    #         for edge_i, edge in enumerate(cp_room_edges):
    #             len_e = len_edge(edge)
    #             if len_e <= 5:
    #                 if len(corner_to_room[edge[0]]) == 1:
    #                     prev_i = edge_i - 1
    #                     prev_edge = cp_room_edges[prev_i]
    #                     cp_room_edges[prev_i] = (prev_edge[0], edge[1])
    #                     rm_flag = True
    #                     cp_room_edges.pop(edge_i)
    #                     break
    #                 elif len(corner_to_room[edge[1]]) == 1:
    #                     next_i = edge_i + 1 if edge_i < len(cp_room_edges) - 1 else 0
    #                     next_edge = cp_room_edges[next_i]
    #                     cp_room_edges[next_i] = (edge[0], next_edge[1])
    #                     rm_flag = True
    #                     cp_room_edges.pop(edge_i)
    #                 else:
    #                     continue
    #
    #     if len(cp_room_edges) != len(room_edges):
    #         all_room_edges[room_idx] = cp_room_edges

    return all_room_edges


def move_corner(c, target, corner_to_room, all_room_edges):
    rooms = corner_to_room[c]
    for room_idx in rooms:
        for edge_idx, edge in enumerate(all_room_edges[room_idx]):
            if c in edge:
                if c == edge[0]:
                    new_edge = (target, edge[1])
                elif c == edge[1]:
                    new_edge = (edge[0], target)
                else:
                    continue
                all_room_edges[room_idx][edge_idx] = new_edge
    return all_room_edges


def find_corners_to_merge(all_corners, corner_to_room, th=5):
    all_close_pairs = list()
    for idx1, corner in enumerate(all_corners):
        for idx2, other_corner in enumerate(all_corners):
            if idx2 <= idx1:
                continue
            if len_edge((corner, other_corner)) <= th:
                rooms_1 = tuple(sorted(corner_to_room[corner]))
                rooms_2 = tuple(sorted(corner_to_room[other_corner]))
                if rooms_1 == rooms_2:
                    continue
                elif len(rooms_1) ==1:
                    if rooms_1[0] in list(rooms_2):
                        continue
                    else:
                        all_close_pairs.append([corner, other_corner])
                elif len(rooms_2) ==1:
                    if rooms_2[0] in list(rooms_1):
                        continue
                    else:
                        all_close_pairs.append([corner, other_corner])
                else:
                    all_close_pairs.append([corner, other_corner])

    if len(all_close_pairs) == 0:
        return None

    close_set = find_one_close_set(all_close_pairs)
    corners_to_merge = list(close_set)

    return corners_to_merge


def find_one_close_set(all_corner_paris):
    all_pairs = list(all_corner_paris)  # make a copy of the input list
    combined = [False] * len(all_corner_paris)

    close_set = _combine_colinear_pairs(0, all_pairs, combined)

    return close_set


def get_corner_to_room(all_room_edges):
    room_paths = [_extract_room_path(room_edges) for room_edges in all_room_edges]
    corner_to_room = defaultdict(list)
    for room_idx, room_path in enumerate(room_paths):
        for corner in room_path:
            corner_to_room[corner].append(room_idx)
    return corner_to_room


def filter_rooms(all_room_edges, im_size):
    # filter rooms that are covered by larger rooms
    room_masks = list()
    updated_room_edges = list()
    for room_edges in all_room_edges:
        room_mask = draw_room_seg_from_edges(room_edges, im_size)
        if room_mask is not None and room_mask.sum() > 20: # remove too small rooms
            room_masks.append(room_mask)
            updated_room_edges.append(room_edges)
    all_room_edges = updated_room_edges

    removed = list()
    for room_idx, room_mask in enumerate(room_masks):
        # do not consider the current room, and do not consider removed rooms
        other_masks = [room_masks[i] for i in range(len(all_room_edges)) if i != room_idx and i not in removed]
        if len(other_masks) == 0:  # if all other masks are removed..
            other_masks_all = np.zeros([im_size, im_size])
        else:
            other_masks_all = np.clip(np.sum(np.stack(other_masks, axis=-1), axis=-1), 0, 1)
        joint_mask = np.clip(other_masks_all + room_mask, 0, 1)
        mask_area = room_mask.sum()
        overlap_area = mask_area + other_masks_all.sum() - joint_mask.sum()
        if overlap_area / mask_area > 0.5:
            removed.append(room_idx)

    all_room_edges = [all_room_edges[idx] for idx in range(len(all_room_edges)) if idx not in removed]

    return all_room_edges


## Utils

class WrongRoomError(Exception):
    pass

def _extract_room_path(room_edges):
    room_path = [edge[0] for edge in room_edges]
    return room_path


def len_edge(e):
    return np.sqrt((e[1][0] - e[0][0]) ** 2 + (e[1][1] - e[0][1]) ** 2)


def unit_v_edge(e):
    len_e = len_edge(e)
    assert len_e != 0
    unit_v = ((e[1][0] - e[0][0]) / len_e, (e[1][1] - e[0][1]) / len_e)
    return unit_v


def get_intersection(p0, p1, p2, p3):
    """
        reference: StackOverflow https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect#565282
    """
    s1_x = p1[0] - p0[0]
    s1_y = p1[1] - p0[1]
    s2_x = p3[0] - p2[0]
    s2_y = p3[1] - p2[1]

    s = (-s1_y * (p0[0] - p2[0]) + s1_x * (p0[1] - p2[1])) / (-s2_x * s1_y + s1_x * s2_y)
    t = (s2_x * (p0[1] - p2[1]) - s2_y * (p0[0] - p2[0])) / (-s2_x * s1_y + s1_x * s2_y)

    if 1 >= s >= 0 and 1 >= t >= 0:
        i_x = p0[0] + (t * s1_x)
        i_y = p0[1] + (t * s1_y)
        return (i_x, i_y)
    else:
        return None


def draw_room_seg_from_edges(edges, im_size):
    edge_map = np.zeros([im_size, im_size])
    for edge in edges:
        edge = np.array(edge).astype(np.int)
        cv2.line(edge_map, tuple(edge[0]), tuple(edge[1]), 1, 3)
    reverse_edge_map = 1 - edge_map
    label, num_features = ndimage.label(reverse_edge_map)
    if num_features < 2:
        return None
    bg_label = label[0, 0]
    num_labels = [(label==l).sum() for l in range(1, num_features+1)]
    num_labels[bg_label-1] = 0
    room_label = np.argmax(num_labels) + 1
    room_map = np.zeros([im_size, im_size])
    room_map[np.where(label == room_label)] = 1

    return room_map



if __name__ == '__main__':
    data_base = './montefloor_data/'
    dir_names = list(sorted(os.listdir(data_base)))

    invalid_scenes = list()

    for dir_name in dir_names:
        if 'scene' not in dir_name:
            continue
        data_dir = os.path.join(data_base, dir_name)
        annot_path = os.path.join(data_dir, 'annotation_3d.json')
        with open(annot_path) as f:
            annot = json.load(f)
        image_path = os.path.join(data_dir, 'density.png')

        try:
            generate_graph(annot, image_path, data_dir)
        except WrongRoomError:
            invalid_scenes.append(dir_name)
        print('Finish processing data {}'.format(dir_name))

    print('Failed on {} scenes with invalid rooms: {}'.format(len(invalid_scenes), invalid_scenes))
