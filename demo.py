from PIL import Image
from predict import annotate_image
from helpers import YoloConfig, download_aerial_dataset
import os
import numpy as np

image_dir, xml_dir = download_aerial_dataset()

imagePath = os.path.join(image_dir,'DJI_0012.jpg')

bbox, score, label = np.array([[232.68, 356.90314, 244.03818, 374.3641]], dtype='float32'), np.array([0.7401025], dtype='float32'), [0]

im = annotate_image(
    imagePath, 
    bbox, score, label)

im.show()