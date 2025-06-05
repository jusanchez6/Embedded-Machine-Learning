
Rock Paper Scissors - v6 EML_96_G_v2
==============================

This dataset was exported via roboflow.com on June 3, 2025 at 9:07 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 7965 images.
Hands are annotated in folder format.

The following pre-processing was applied to each image:
* Resize to 96x96 (Fit within)
* Grayscale (CRT phosphor)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Randomly crop between 0 and 20 percent of the image


