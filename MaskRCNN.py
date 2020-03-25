from google.colab import drive
drive.mount('/content/drive')

import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
import imgaug
# import skimage.draw
from matplotlib import pyplot as plt
# Root directory of the project
ROOT_DIR = ('/content/drive/My Drive/GarudaUAV/Mask_RCNN')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class GarudaConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "garuda"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + objects

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    BACKBONE = 'resnet50'
    # Skip detections with < 90% confidence
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64, 128)
    # TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    # POST_NMS_ROIS_INFERENCE = 500 
    # POST_NMS_ROIS_TRAINING = 1000 
    LOSS_WEIGHTS = {
          "rpn_class_loss": 1.0, # How well the Region Proposal Network separates background with objetcs
          "rpn_bbox_loss": 0.7, # How well the RPN localize objects
          "mrcnn_class_loss": 1.0, # How well the Mask RCNN localize objects
          "mrcnn_bbox_loss": 1.0, # How well the Mask RCNN recognize each class of object
          "mrcnn_mask_loss": 1.0 # How well the Mask RCNN segment objects
    }


config = GarudaConfig()
config.display()


class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """
    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        # Add the class names using the base method from utils.Dataset
        source_name = "garuda"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                try:
                  image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                  image_annotations = annotations[image_id]
                  
                  # Add the image using the base method from utils.Dataset
                  self.add_image(
                      source=source_name,
                      image_id=image_id,
                      path=image_path,
                      width=image_width,
                      height=image_height,
                      annotations=image_annotations
                  )
                except KeyError as key1:
                  print('Id not found. Skipping')
    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids


dataset_train = CocoLikeDataset()
# dataset_train.load_data('/content/drive/My Drive/GarudaUAV/Garuda/Train_Images/traincoco.json', '/content/drive/My Drive/GarudaUAV/Garuda/Train_Images')
dataset_train.load_data('/content/drive/My Drive/GarudaUAV/Garuda/fromlabelme/t1/trainval.json', '/content/drive/My Drive/GarudaUAV/Garuda/fromlabelme/t1')
dataset_train.prepare()

dataset_val = CocoLikeDataset()
# dataset_val.load_data('/content/drive/My Drive/GarudaUAV/Garuda/Val_Images/valcoco.json', '/content/drive/My Drive/GarudaUAV/Garuda/Val_Images/')
dataset_val.load_data('/content/drive/My Drive/GarudaUAV/Garuda/fromlabelme/v1/trainval.json', '/content/drive/My Drive/GarudaUAV/Garuda/fromlabelme/v1')
dataset_val.prepare()


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


class_names = ['BG', 'helmet', 'jacket', 'Person']



# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


augmentation = imgaug.augmenters.Fliplr(0.5)
start_train = time.time()
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=5, 
            layers='heads',
            augmentation=augmentation)
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f'Training took {minutes} minutes')



augmentation = imgaug.augmenters.Fliplr(0.5)
start_train = time.time()
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=5, 
            layers='heads',
            augmentation=augmentation)
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f'Training took {minutes} minutes')


class InferenceConfig(GarudaConfig):
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.85
    

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

model_path = model.find_last()

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
# model_path = '/content/drive/My Drive/GarudaUAV/Mask_RCNN/logs/garuda20200310T2145/mask_rcnn_garuda_0004.h5'
model.load_weights(model_path, by_name=True)

import skimage
real_test_dir = '/content/drive/My Drive/GarudaUAV/Garuda/test'
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(real_test_dir, filename))

for image_path in image_paths:
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=1)
    r = results[0]
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                dataset_val.class_names, r['scores'], figsize=(15,15))