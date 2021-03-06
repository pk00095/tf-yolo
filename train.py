"""
Retrain the YOLO model for your own dataset.
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import yolo_body, tiny_yolo_body, yolo_loss, Yolo_Loss
from helpers import YoloConfig, download_aerial_dataset
from tfrecord_creator import create_tfrecords

from object_detection_parser import DetectionBase
# from yolo3.utils import get_random_data


# def _main():
#     annotation_path = 'train.txt'
#     log_dir = 'logs/000/'
#     classes_path = 'model_data/voc_classes.txt'
#     anchors_path = 'model_data/yolo_anchors.txt'
#     class_names = get_classes(classes_path)
#     num_classes = len(class_names)
#     anchors = get_anchors(anchors_path)

#     input_shape = (416,416) # multiple of 32, hw

#     is_tiny_version = len(anchors)==6 # default setting
#     if is_tiny_version:
#         model = create_tiny_model(input_shape, anchors, num_classes,
#             freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
#     else:
#         model = create_model(input_shape, anchors, num_classes,
#             freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze

#     logging = TensorBoard(log_dir=log_dir)
#     checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
#         monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
#     early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

#     val_split = 0.1
#     with open(annotation_path) as f:
#         lines = f.readlines()
#     np.random.seed(10101)
#     np.random.shuffle(lines)
#     np.random.seed(None)
#     num_val = int(len(lines)*val_split)
#     num_train = len(lines) - num_val

#     # Train with frozen layers first, to get a stable loss.
#     # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
#     if True:
#         model.compile(optimizer=Adam(lr=1e-3), loss={
#             # use custom yolo_loss Lambda layer.
#             'yolo_loss': lambda y_true, y_pred: y_pred})

#         batch_size = 32
#         print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
#         model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
#                 steps_per_epoch=max(1, num_train//batch_size),
#                 validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
#                 validation_steps=max(1, num_val//batch_size),
#                 epochs=50,
#                 initial_epoch=0,
#                 callbacks=[logging, checkpoint])
#         model.save_weights(log_dir + 'trained_weights_stage_1.h5')

#     # Unfreeze and continue training, to fine-tune.
#     # Train longer if the result is not good.
#     if True:
#         for i in range(len(model.layers)):
#             model.layers[i].trainable = True
#         model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
#         print('Unfreeze all of the layers.')

#         batch_size = 32 # note that more GPU memory is required after unfreezing the body
#         print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
#         model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
#             steps_per_epoch=max(1, num_train//batch_size),
#             validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
#             validation_steps=max(1, num_val//batch_size),
#             epochs=100,
#             initial_epoch=50,
#             callbacks=[logging, checkpoint, reduce_lr, early_stopping])
#         model.save_weights(log_dir + 'trained_weights_final.h5')


def create_model(config, num_classes, load_pretrained=True, freeze_body=2):
    '''create the training model'''
    K.clear_session() # get a new session
    input_shape = config.input_shape
    image_input = Input(shape=(input_shape[0], input_shape[1], 3))
    # image_input = Input(shape=(None, None, 3))
    # h, w = config.input_shape
    num_anchors = len(config.anchors)

    # y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
    #     num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        # weights_path = keras.utils.get_file(
        #     fname='darknet53_notop_weights.h5', 
        #     origin='https://drive.google.com/u/0/uc?export=download&confirm=uCPi&id=1RwvRnB-t2x-LMhU9oKcuJmKzCXInSsUw')
        weights_path = './pretrained/darknet53.h5'
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    # model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
    #     arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
    #     [*model_body.output, *y_true])
    # model = Model([model_body.input, *y_true], model_loss)

    return model_body

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def main():
    import os 

    image_dir, xml_dir = download_aerial_dataset()
    create_tfrecords(image_dir, xml_dir)
    
    config = YoloConfig()

    num_classes = 4
    yolo_model = create_model(config=config, num_classes=num_classes, load_pretrained=False, freeze_body=1) #freeze_body=1 means only freeze the backbone

    dataset_func = DetectionBase(
        train_tfrecords=os.path.join(os.getcwd(), 'DATA' ,'train*.tfrecord'),
        test_tfrecords=os.path.join(os.getcwd(), 'DATA' ,'test*.tfrecord'), 
        num_classes=num_classes, 
        config=config,
        batch_size=1)

    training_dataset = dataset_func.get_train_function()

    l1_candidate_anchors = config.anchors[config.anchor_mask[0]]
    l2_candidate_anchors = config.anchors[config.anchor_mask[1]]
    l3_candidate_anchors = config.anchors[config.anchor_mask[2]]

    layer1_loss = Yolo_Loss(input_shape=config.input_shape, candidate_anchors=l1_candidate_anchors, grid_shape=yolo_model.outputs[0].shape, num_classes=num_classes)
    layer2_loss = Yolo_Loss(input_shape=config.input_shape, candidate_anchors=l2_candidate_anchors, grid_shape=yolo_model.outputs[1].shape, num_classes=num_classes)
    layer3_loss = Yolo_Loss(input_shape=config.input_shape, candidate_anchors=l3_candidate_anchors, grid_shape=yolo_model.outputs[2].shape, num_classes=num_classes)

    yolo_model.compile(
        optimizer=Adam(lr=1e-3),
        loss={
        'tf_op_layer_y1_pred':layer1_loss,
        'tf_op_layer_y2_pred':layer2_loss,
        'tf_op_layer_y3_pred':layer3_loss}
            )

    yolo_model.fit(training_dataset, epochs=10, steps_per_epoch=100)



if __name__ == '__main__':
    main()
