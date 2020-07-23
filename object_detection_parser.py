# from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras
import glob, os
# import pandas as pd
import numpy as np
import cv2
# from albumentations import Compose, BboxParams

def print_annotations_dims(x):
    tf.print(x.shape)
    # tf.print(bboxes)
    return x

def _pad(image, height, width):
    """Summary
    
    Args:
        image (TYPE): Description
        height (TYPE): Description
        width (TYPE): Description
        scale (TYPE): Description
    
    Returns:
        numpy nd.array: Description
    """

    image = image.astype(np.uint8)
    padded_image = np.zeros(shape=(height.astype(int), width.astype(int),3), dtype=np.uint8)
    h,w,_ =  image.shape
    padded_image[:h,:w,:] = image
    return padded_image

@tf.function
def decode_pad(image_string, pad_height, pad_width):
  """Summary
  
  Args:
      image_string (TYPE): Description
      pad_height (TYPE): Description
      pad_width (TYPE): Description
      scale (TYPE): Description
  
  Returns:
      tf.tensor: Description
  """
  image = tf.image.decode_jpeg(image_string)
  image = tf.numpy_function(_pad, [image, pad_height, pad_width], Tout=tf.uint8)
  #image.set_shape([None, None, 3])
  return image



class DetectionBase(object):
    """docstring for DetectionBase"""
    def __init__(self, 
        train_tfrecords,
        test_tfrecords, 
        num_classes, 
        config,
        batch_size=4):

        self.train_tfrecords = train_tfrecords
        self.test_tfrecords = test_tfrecords

        self.num_classes = int(num_classes)
        self.batch_size = batch_size
        # self.aug = augmentation
        # self.bboxes_format = bboxes_format
        self.config = config

    def preprocess_true_boxes(self, true_boxes):
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
        assert (true_boxes[..., 4]<self.num_classes).all(), 'class id must be less than num_classes'
        num_layers = self.config.num_layers # default setting
        anchor_mask = self.config.anchor_mask#[[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(self.config.input_shape, dtype='int32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

        m = true_boxes.shape[0]
        grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
        y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+self.num_classes),
            dtype='float32') for l in range(num_layers)]

        # Expand dim to apply broadcasting.
        anchors = np.expand_dims(self.config.anchors, 0)
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

        # print('--------')
        # print(len(y_true))
        # for y in y_true:
        #   print(y.shape, y.dtype)
        return y_true


    def tf_preprocess_true_boxes(self, true_boxes):
      # because num layersare hard coded for 3 now
      y_true_layer1, y_true_layer2, y_true_layer3 = tf.numpy_function(self.preprocess_true_boxes, [true_boxes], Tout=[keras.backend.floatx()]*self.config.num_layers, name='yolo_to_gt')
      return y_true_layer1, y_true_layer2, y_true_layer3

    def pascalvoc_to_yolo(self, image_array, box):
        # print(image_array.shape, box.shape)
        ih, iw, _c = image_array.shape
        h, w = self.config.input_shape
        box = box[~np.all(box<0, axis=-1)] #remove all -1

        # if not random:
            # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0

        # if proc_img:
        #     image = image.resize((nw,nh), Image.BICUBIC)
        #     new_image = Image.new('RGB', (w,h), (128,128,128))
        #     new_image.paste(image, (dx, dy))
        #     image_data = np.array(new_image)/255.

        image = cv2.resize(image_array, (nw, nh), interpolation=cv2.INTER_CUBIC)
        new_image = np.ones(shape=(h,w,3), dtype=np.uint8)*128
        new_image[dy:dy+nh,dx:dx+nw,:] = image
        image_data = new_image /255.

        # correct boxes
        box_data = np.zeros((self.config.max_boxes,5), dtype=keras.backend.floatx())
        if len(box)>0:
            np.random.shuffle(box)
            # if number boxes exceeds number of expected detection, remove bboxes from last
            if len(box)>self.config.max_boxes: 
                box = box[:self.config.max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box
        # tf.print(box.shape)
        return image_data.astype(keras.backend.floatx()), box_data.astype(keras.backend.floatx())

    def tf_pascalvoc_to_yolo(self, image_batch, xmin_batch, ymin_batch, xmax_batch, ymax_batch, label_batch):

        image_data = list()
        annbox_data = list()

        for index in range(self.batch_size):
            image = image_batch[index]
            xmins, ymins, xmaxs, ymaxs, labels = xmin_batch[index], ymin_batch[index], xmax_batch[index], ymax_batch[index], label_batch[index]
            bboxes = tf.convert_to_tensor([xmins,ymins,xmaxs,ymaxs, labels], dtype=tf.keras.backend.floatx())
            bboxes = tf.transpose(bboxes)

            image_preprocessed, annotation_box = tf.numpy_function(self.pascalvoc_to_yolo, [image, bboxes], Tout=[keras.backend.floatx(), keras.backend.floatx()], name='pascalvoc_to_yolo')

            image_data.append(image_preprocessed)
            annbox_data.append(annotation_box)

        return tf.convert_to_tensor(image_data), tf.convert_to_tensor(annbox_data)



    def _parse_function(self, serialized):

        features = {
          'image/height': tf.io.FixedLenFeature([], tf.int64),
          'image/width': tf.io.FixedLenFeature([], tf.int64),
          'image/encoded': tf.io.FixedLenFeature([],tf.string),
          'image/object/bbox/xmin': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/object/bbox/xmax': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/object/bbox/ymin': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/object/bbox/ymax': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/f_id': tf.io.FixedLenFeature([], tf.int64),
          'image/object/class/label':tf.io.VarLenFeature(tf.int64)}


        parsed_example = tf.io.parse_example(serialized=serialized, features=features)

        max_height = tf.cast(tf.keras.backend.max(parsed_example['image/height']), tf.int32)
        max_width = tf.cast(tf.keras.backend.max(parsed_example['image/width']), tf.int32)

        image_batch = tf.map_fn(lambda x: decode_pad(x, max_height, max_width), parsed_example['image/encoded'], dtype=tf.uint8)

        # Each of the following has batch (batch_size, 1, max_boxes_in_batch)
        xmin_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/xmin'], default_value=-1)
        # xmin_batch.set_shape([None,None])

        xmax_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/xmax'], default_value=-1)
        # xmax_batch.set_shape([None,None])

        ymin_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/ymin'], default_value=-1)
        # ymin_batch.set_shape([None,None])

        ymax_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'], default_value=-1)
        # ymax_batch.set_shape([None,None])

        label_batch = tf.cast(tf.sparse.to_dense(parsed_example['image/object/class/label'], default_value=-1), keras.backend.floatx())
        # label_batch.set_shape([None,None])
        # dimension is (batch_size, 5, max_boxes_in_batch)
        #boxes_batch = tf.concat([xmin_batch, ymin_batch, xmax_batch, ymax_batch, label_batch], axis=-1)
        # boxes_batch.set_shape([None, None, None])
        # dimension is (batch_size, max_boxes_in_batch, 5)
        #boxes_batch = tf.transpose(boxes_batch, perm=[1,0])


        im_batch, annotation_batch = self.tf_pascalvoc_to_yolo(image_batch, xmin_batch, ymin_batch, xmax_batch, ymax_batch, label_batch)

        y_true_layer1, y_true_layer2, y_true_layer3 = self.tf_preprocess_true_boxes(annotation_batch)

        # return im_batch, annotation_batch
        return im_batch, {'y1_pred':y_true_layer1, 'y2_pred':y_true_layer2, 'y3_pred':y_true_layer3}



    def get_train_function(self):

        dataset = tf.data.Dataset.list_files(self.train_tfrecords).shuffle(buffer_size=256).repeat(-1)
        dataset = dataset.interleave(
          tf.data.TFRecordDataset, 
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          deterministic=False)

        dataset = dataset.batch(
          self.batch_size, 
          drop_remainder=True)    # Batch Size

        dataset = dataset.map(
          self._parse_function, 
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # dataset = dataset.cache()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def get_test_function(self):

        dataset = tf.data.Dataset.list_files(self.test_tfrecords).shuffle(buffer_size=256).repeat(-1)
        dataset = dataset.interleave(
          tf.data.TFRecordDataset, 
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          deterministic=False)

        dataset = dataset.batch(
          self.batch_size, 
          drop_remainder=True)    # Batch Size

        dataset = dataset.map(
          self._parse_function, 
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # dataset = dataset.cache()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset


if __name__ == '__main__':
  from helpers import YoloConfig

  config = YoloConfig()

  dataset_func = DetectionBase(
        train_tfrecords=os.path.join(os.getcwd(), 'DATA' ,'train*.tfrecord'),
        test_tfrecords=os.path.join(os.getcwd(), 'DATA' ,'test*.tfrecord'), 
        num_classes=5, 
        config=config)

  training_dataset = dataset_func.get_train_function()

  for im_batch, y1, y2,y3 in training_dataset.take(6):
    print(im_batch.shape, y1.shape, y2.shape, y3.shape)
