

class HyperParams(object):

    def __init__(self):
        self.min_img_width = 300

        # anchor box scales
        self.anchor_box_scales = [128, 256, 512]
        # anchor box ratios
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
        self.num_anchor = len(self.anchor_box_ratios) * len(self.anchor_box_scales)

        self.down_scale = 16

        self.rpn_num_regions = 256
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        self.std_scaling = 4.0

H = HyperParams()
