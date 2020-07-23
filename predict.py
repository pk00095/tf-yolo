import tensorflow as tf
from tensorflow import keras
from helpers import YoloConfig
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body


def freeze_model(model_path, config, num_classes, max_boxes=20, score_threshold=.6,iou_threshold=.5):
    # model_path = os.path.expanduser(self.model_path)
    # assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

    # Load model, or construct model and load weights.
    num_anchors = len(config.anchors)
    # num_classes = len(self.class_names)
    # is_tiny_version = num_anchors==6 # default setting
    yolo_model = keras.models.load_model(model_path, compile=False)

    boxes, scores, classes = yolo_eval(
    	yolo_model.outputs, 
    	config.anchors,
        num_classes, 
        config.input_shape,
        anchor_mask=config.anchor_mask,
        max_boxes=config.max_boxes,
        score_threshold=score_threshold, 
        iou_threshold=iou_threshold)

    prediction_model = keras.models.Model(yolo_model.input, [boxes, scores, classes])

    return boxes, scores, classes