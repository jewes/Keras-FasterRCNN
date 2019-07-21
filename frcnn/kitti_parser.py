import os
import tqdm


class KittiParser(object):

    def __init__(self, data_dir):
        """

        :param data_dir: the data_dir should contains 2 folders, a labels folder and images folder
        """
        self.labels_dir = os.path.join(data_dir, "labels")
        self.images_dir = os.path.join(data_dir, "images")

        if not os.path.exists(self.labels_dir) or not os.path.exists(self.images_dir):
            raise IOError('missing labels or images folder')

    def get_annotations(self):
        label_files = os.listdir(self.labels_dir)
        if len(label_files) == 0:
            raise IOError('no label files found.')

        pbar = tqdm.tqdm(total=len(label_files))
        annotations = []

        for label_file in label_files:
            pbar.update(1)
            filename, ext = os.path.splitext(label_file)
            image_path = os.path.join(self.images_dir, "{}.jpg".format(filename))
            if os.path.exists(image_path):
                print("Warn: no corresponding images at {}".format(image_path))
                continue

            bboxes = []
            label_file_path = os.path.join(self.labels_dir, label_file)
            with os.open(label_file_path) as fp:
                for line in fp:
                    segments = line.strip().split()
                    category = segments[0]
                    xmin = segments[4]
                    ymin = segments[5]
                    xmax = segments[6]
                    ymax = segments[7]
                    bboxes.append({
                        'class': category,
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax
                    })
                if len(bboxes) > 0:
                    annotations.append({
                        'file_path': image_path,
                        'bboxes': bboxes
                    })

        return annotations
