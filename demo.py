from PIL import Image
from predict import annotate_image, freeze_model, detect_image
from helpers import YoloConfig, download_aerial_dataset
import os
import numpy as np
# from tensorflow.keras import backend as K
# K.clear_session()

from PIL import Image

config = YoloConfig()
num_classes = 4
prediction_model = freeze_model('aerial_yolo', config, num_classes)
image_dir, xml_dir = download_aerial_dataset()

imagePath = os.path.join(image_dir,'DJI_0012.jpg')

bbox, score, label = detect_image(
    model=prediction_model, 
    image=Image.open(imagePath), 
    config=config)

im = annotate_image(
    imagePath, 
    bbox, score, label)

im = im.resize((640,480))
im.save('annotated.jpeg')

