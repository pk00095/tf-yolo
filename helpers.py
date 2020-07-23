import tensorflow as tf
import zipfile, tempfile, os

import numpy as np
from utils import get_random_data


class YoloConfig(object):
    """docstring for YoloConfig"""
    def __init__(self, 
        anchors=[
                (10,13),
                (16,30),
                (33,23),
                (30,61),
                (62,45),
                (59,119),
                (116,90),
                (156,198),
                (373,326)],
        anchor_mask=[
                [6,7,8], 
                [3,4,5], 
                [0,1,2]],
        max_boxes=20, 
        height=416, 
        width=416,
        score=0.3,
        iou=0.45):
        # super(YoloConfig, self).__init__()
        assert height%32 == 0
        assert width%32 == 0

        assert isinstance(anchors, list)

        self.anchors = np.array(anchors, dtype='float32').reshape(-1, 2)

        assert self.anchors.shape[0]%3==0

        self.height = height
        self.width = width

        self.input_shape = (self.height, self.width)

        self.anchor_mask = anchor_mask
        self.max_boxes = max_boxes

        self.num_layers = len(self.anchors)//3  #default setting

        assert self.num_layers == 3, 'currently support only 3 pyramid layers' #have tomake it dynamic


def download_aerial_dataset(dataset_path=tempfile.gettempdir()):
    zip_url = 'https://segmind-data.s3.ap-south-1.amazonaws.com/edge/data/aerial-vehicles-dataset.zip'
    path_to_zip_file = tf.keras.utils.get_file(
        'aerial-vehicles-dataset.zip',
        zip_url,
        cache_dir=dataset_path, 
        cache_subdir='',
        extract=False)
    directory_to_extract_to = os.path.join(dataset_path,'aerial-vehicles-dataset')
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    images_dir = os.path.join(dataset_path, 'aerial-vehicles-dataset','images')
    annotation_dir = os.path.join(dataset_path, 'aerial-vehicles-dataset','annotations','pascalvoc_xml')

    return images_dir, annotation_dir

def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes, anchor_mask):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = anchor_mask#[[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true


def data_generator(annotation_path, batch_size, input_shape, anchors, num_classes,anchor_mask):
    '''data generator for fit_generator'''
    with open(annotation_path) as f:
        annotation_lines = f.readlines()
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=False)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes, anchor_mask)
        yield [image_data, *y_true], np.zeros(batch_size)

if __name__ == '__main__':
	batch_size = 4
	annotation_path ='/home/pratik/Desktop/experiments/PLATFORM/yolo_experiments/aerial.txt'

	config=YoloConfig()

	for x,y in data_generator(annotation_path, batch_size,  config.input_shape, config.anchors, num_classes=5, anchor_mask=config.anchor_mask):

		print(x[0].shape, x[1].shape)

		# print(x, y)