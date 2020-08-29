# tf-yolo

## Introduction
an implementation of yolov3 with tensorflow-2.2, which uses tf.data api on tfrecords to feed data. Which makes training faster through data parallelism

## Custom training
- convert data in pascal-voc xml format to tfrecords using `tfrecord_creator.py`. This will create a folder `DATA` and store the tfrecords in 4 shards there.
- customize `main.py` according to your dataset and start training

## Prediction
- customize `demo.py` to predict on a new image

## TO DO
- Provide a cli for training
- Predict on multiple images
- Add tensorboard summary
- Implement Tiny-YOLO

## Reference
@article{redmon2018yolov3,
  title={Yolov3: An incremental improvement. arXiv 2018},
  author={Redmon, Joseph and Farhadi, Ali},
  journal={arXiv preprint arXiv:1804.02767},
  pages={1--6},
  year={2018}
}




