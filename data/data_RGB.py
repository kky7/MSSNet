import os
from . import dataset_RGB

def get_training_data(image_list, root_dir, img_options):
    return dataset_RGB.DataLoaderTrain(image_list,root_dir,img_options)

def get_validation_data(rgb_dir, root_dir, img_options):
    return dataset_RGB.DataLoaderVal(rgb_dir, root_dir, img_options)

def get_test_data(rgb_dir, root_dir):
    return dataset_RGB.DataLoaderTest(rgb_dir, root_dir)
