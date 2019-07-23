import random

import cv2
import numpy as np


# image resize
from keras.applications import imagenet_utils
from .kitti_parser import KittiParser
from .hyper_params import H


def get_new_img_size(width, height, img_min_side):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height


# Intersection of Union
def iou(a, b):
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


class TrainDataGenerator(object):

    def __init__(self, data_dir, annotation_format='kitti'):
        self.annotation_data = None
        if 'kitti' == annotation_format:
            annotation_parser = KittiParser(data_dir)
            self.annotation_data = annotation_parser.get_annotations()

        if self.annotation_data is None:
            raise ValueError('missing annotation')

    def get_train_datagen(self):
        while True:
            # shuffle it in every epoch
            random.shuffle(self.annotation_data)

            # ensure every image is used for training at least once.
            for img_data in self.annotation_data:
                file_path = img_data.get('file_path')
                print("reading image {}".format(file_path))

                x_img = cv2.imread(file_path)
                if x_img is None:
                    print("error: reading image {} failed.".format(file_path))
                    continue

                w, h = x_img.shape[:2]
                resized_w, resized_h = get_new_img_size(w, h, img_min_side=H.min_img_width)

                # prepare X
                x_img = cv2.resize(x_img, (resized_w, resized_h))
                x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
                x_img = np.expand_dims(x_img, axis=0)
                x_img = imagenet_utils.preprocess_input(x_img, mode='tf')

                # prepare Y
                # now calculates the rpn gt cls and regression box
                y_rpn_cls, y_rpn_regr = self.calc_rpn_gt(resized_w, resized_h, w, h, img_data['bboxes'])

                y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= 4.0  # scaling the regression target, i don't know why

                # change shape to NWHC format
                y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                yield x_img, [y_rpn_cls, y_rpn_regr]

    def calc_rpn_gt(self, resized_width, resized_height, w, h, gt_bboxes,
                    rpn_min_overlap=H.rpn_min_overlap, rpn_max_overlap=H.rpn_max_overlap, num_region=H.rpn_num_regions):
        anchor_sizes = H.anchor_box_scales
        anchor_ratios = H.anchor_box_ratios
        num_anchors = len(anchor_sizes) * len(anchor_ratios)

        output_width, output_height = int(resized_width / H.down_scale), int(resized_height / H.down_scale)

        num_bbox = len(gt_bboxes)
        gta = np.zeros((num_bbox, 4))  # store the gt bboxes
        for idx, bbox in enumerate(gt_bboxes):
            gta[idx, 0] = bbox['xmin'] * (resized_width / float(w))
            gta[idx, 1] = bbox['xmax'] * (resized_width / float(w))
            gta[idx, 2] = bbox['ymin'] * (resized_height / float(h))
            gta[idx, 3] = bbox['ymax'] * (resized_height / float(h))

        # from anchor perspective
        y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))  # 记录iou？
        y_is_box_valid = np.zeros((output_height, output_width, num_anchors))  # objectness?
        y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))  # anchor bbox

        # from the gt bbox perspective.
        bbox_num_pos_anchors = np.zeros(num_bbox).astype(int)
        best_anchor_for_bbox = -1 * np.ones((num_bbox, 4)).astype(int)
        best_iou_for_bbox = np.zeros(num_bbox).astype(np.float32)
        best_x_for_bbox = np.zeros((num_bbox, 4)).astype(int)
        best_dx_for_bbox = np.zeros((num_bbox, 4)).astype(np.float32)

        # step 1:
        # for all the feature map locations, calculates its iou with the gt boxes, and determine its training target
        # i.e. whether it should predict an object or background
        for idx_anchor_size, anchor_size in enumerate(anchor_sizes):
            for idx_anchor_ratio, anchor_ratio in enumerate(anchor_ratios):
                anchor_w = anchor_size * anchor_ratio[0]
                anchor_h = anchor_size * anchor_ratio[1]

                for fx in range(output_width):
                    # self.down_scale * (fx + 0.5) is the center_x of the anchor box
                    anchor_xmin = H.down_scale * (fx + 0.5) - anchor_w / 2
                    anchor_xmax = H.down_scale * (fx + 0.5) + anchor_w / 2

                    # anchor xmin or xmax exceeds the image
                    if anchor_xmin < 0 or anchor_xmax > resized_width:
                        continue
                    for fy in range(output_height):
                        anchor_ymin = H.down_scale * (fy + 0.5) - anchor_h / 2
                        anchor_ymax = H.down_scale * (fy + 0.5) + anchor_h / 2

                        if anchor_ymin < 0 or anchor_ymax > resized_height:
                            continue

                        # this is the best IOU for the (x,y) coord and the current anchor
                        # note that this is different from the best IOU for a GT bbox
                        best_iou_for_current_location = 0.0
                        best_regr = [0.0] * 4
                        # now we get the anchor coordinate, let's calculate the IOU
                        anchor_type = 'neg'
                        for idx_bbox in range(len(gt_bboxes)):
                            gt_xmin, gt_xmax, gt_ymin, gt_ymax = gta[idx_bbox, 0], gta[idx_bbox, 1], gta[idx_bbox, 2], \
                                                                 gta[idx_bbox, 3]

                            # the iou between the current anchor and the gt box
                            current_iou = iou([gt_xmin, gt_xmax, gt_ymin, gt_ymax],
                                              [anchor_xmin, anchor_xmax, anchor_ymin, anchor_ymax])

                            # find a good match anchor box
                            if current_iou > best_iou_for_bbox[idx_bbox] or current_iou > rpn_max_overlap:
                                # calculate the deltas
                                gt_cx = (gt_xmin + gt_xmax) / 2.0
                                gt_cy = (gt_ymin + gt_ymax) / 2.0
                                anchor_cx = (anchor_xmin + anchor_xmax) / 2.0
                                anchor_cy = (anchor_ymin + anchor_ymax) / 2.0

                                delta_cx = (gt_cx - anchor_cx) / anchor_w
                                delta_cy = (gt_cy - anchor_cy) / anchor_h

                                delta_w = np.log((gt_xmax - gt_xmin) / anchor_w)
                                delta_h = np.log((gt_ymax - gt_ymin) / anchor_h)

                                # setting the gt box related variables
                                # each GT boxes should be mapped to an anchor box,
                                # so we keep track of which anchor box was best
                                if current_iou > best_iou_for_bbox[idx_bbox]:
                                    best_anchor_for_bbox[idx_bbox, :] = [fy, fx, idx_anchor_ratio, idx_anchor_size]
                                    best_iou_for_bbox[idx_bbox] = current_iou
                                    best_x_for_bbox[idx_bbox, :] = [anchor_xmin, anchor_xmax, gt_xmin, gt_xmax]
                                    best_dx_for_bbox[idx_bbox, :] = [delta_cx, delta_cy, delta_w, delta_h]

                                if current_iou > rpn_max_overlap:
                                    bbox_num_pos_anchors[idx_bbox] += 1
                                    anchor_type = 'pos'

                                    if current_iou > best_iou_for_current_location:
                                        best_iou_for_current_location = current_iou
                                        best_regr = [delta_cx, delta_cy, delta_w, delta_h]

                                if rpn_min_overlap < current_iou < rpn_max_overlap:
                                    # gray zone between neg and pos
                                    if anchor_type != 'pos':
                                        anchor_type = 'neutral'

                        # now all the gt_boxes have been processed for one anchor location
                        idx_anchor = idx_anchor_size * len(anchor_ratios) + idx_anchor_ratio
                        if anchor_type == 'neg':
                            y_is_box_valid[fy, fx, idx_anchor] = 1
                            y_rpn_overlap[fy, fx, idx_anchor] = 0
                        elif anchor_type == 'neutral':
                            y_is_box_valid[fy, fx, idx_anchor] = 0
                            y_rpn_overlap[fy, fx, idx_anchor] = 0
                        elif anchor_type == 'pos':
                            y_is_box_valid[fy, fx, idx_anchor] = 1
                            y_rpn_overlap[fy, fx, idx_anchor] = 1
                            start = 4 * idx_anchor
                            y_rpn_regr[fy, fx, start:start + 4] = best_regr

        # we processed all the locations, check if any bbox has no anchor covered
        for idx in range(num_bbox):
            if bbox_num_pos_anchors[idx] == 0:
                if best_anchor_for_bbox[idx, 0] == -1:
                    print('Warning: no overlap anchor for bbox {}'.format(idx))
                    continue

                second_best_anchor_for_bbox = best_anchor_for_bbox[idx]
                fy = second_best_anchor_for_bbox[0],
                fx = second_best_anchor_for_bbox[1],
                idx_anchor_ratio = second_best_anchor_for_bbox[2],
                idx_anchor_size = second_best_anchor_for_bbox[3]

                idx_anchor = len(anchor_ratios) * idx_anchor_size + idx_anchor_ratio
                y_is_box_valid[fy, fx, idx_anchor] = 1
                y_rpn_overlap[fy, fx, idx_anchor] = 1
                start = 4 * idx_anchor
                y_rpn_regr[fy, fx, start:start + 4] = best_dx_for_bbox[idx, :]

        # now - each location has been marked and each bbox has been associated with an anchor
        # next, we will select a few negative sample ans the positive samples to form mini batch training samples
        # move the anchor-axis to the first dimension
        y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
        y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)  # 0-dimension is the batch dimension

        y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
        y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

        y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
        y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

        # return 3 sub arrays, the indexes of the element meet the condition
        pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
        neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

        num_positive_locs = len(pos_locs[0])
        num_negative_locs = len(neg_locs[0])

        # mute some positive box
        if num_positive_locs > num_region / 2:
            val_locs = random.sample(num_positive_locs, num_positive_locs - num_region / 2)
            y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
            num_positive_locs = num_region / 2

        # has more negative locations, then mute some negative anchors
        if num_negative_locs + num_positive_locs > num_region:
            val_locs = random.sample(range(num_negative_locs), num_negative_locs - num_positive_locs)  # neg:pos = 1:1
            y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

        # y_rpn_cls shape: (1, 18, rows, cols), 1st dimension: is_box_valid, y_rpn_overlap for each anchor
        # merge them together, which will be used in computing loss.
        # y_true and y_pred is not 1 vs 1 mapping
        y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)

        # y_rpn_regr shape: (1, 72, rows, cols)?
        y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

        return np.copy(y_rpn_cls), np.copy(y_rpn_regr)

