import os
import tqdm
import cv2
import numpy as np

def get_data(input_path):
    labels_dir = os.path.join(input_path, "labels")
    images_dir = os.path.join(input_path, "images")

    if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
        raise IOError('missing labels or images folder')

    label_files = os.listdir(labels_dir)
    if len(label_files) == 0:
        raise IOError('no label files found.')

    pbar = tqdm.tqdm(total=len(label_files))
    all_imgs = {}
    classes_count = {}
    classes_mapping = {}

    for label_file in label_files:
        pbar.update(1)
        filename, ext = os.path.splitext(label_file)
        image_path = os.path.join(images_dir, "{}.jpg".format(filename))
        if not os.path.exists(image_path):
            print("Warn: no corresponding images at {}".format(image_path))
            continue

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        all_imgs[filename] = {}
        all_imgs[filename]['filepath'] = image_path
        all_imgs[filename]['width'] = w
        all_imgs[filename]['height'] = h
        all_imgs[filename]['bboxes'] = []
        if np.random.randint(0, 6) > 0:
            all_imgs[filename]['imageset'] = 'trainval'
        else:
            all_imgs[filename]['imageset'] = 'test'

        label_file_path = os.path.join(labels_dir, label_file)
        with open(label_file_path, 'r') as fp:
            for line in fp:
                segments = line.strip().split()
                category = segments[0]

                if category not in classes_count:
                    classes_count[category] = 1
                else:
                    classes_count[category] += 1

                if category not in classes_mapping:
                    classes_mapping[category] = len(classes_mapping)

                xmin = segments[4]
                ymin = segments[5]
                xmax = segments[6]
                ymax = segments[7]
                all_imgs[filename]['bboxes'].append({
                    'class': category,
                    'x1': int(xmin),
                    'y1': int(ymin),
                    'x2': int(xmax),
                    'y2': int(ymax)
                })
    all_data = []
    for key in all_imgs:
        all_data.append(all_imgs[key])

    return all_data, classes_count, classes_mapping


if __name__ == '__main__':
    all_data, classes_count, classes_mapping = get_data("../dataset")
    pass
