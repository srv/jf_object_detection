# Jellyfish object detection

This repository aims to provide the necessary tools and knowledge to operate our jellyfish object detection and monitoring algorithm. The following sections provide a step by step explanation on how to:

1 - Train and infer an object detection neural network.

2 - Optimize and evaluate the obtained detections.

3 - Apply quantification and monitoring algorithm

# 1 - Train and infer an object detection neural network.

To train an object detection neural network from scratch, we refer to the Google object detection API, which uses Tensorflow: https://github.com/tensorflow/models/tree/196d173a24613a045e641ef21ba9863c77bd1e2f/research/object_detection.

We provide our training data in the following link: https://zenodo.org/record/3537652#.XclltcYh3CI

Otherwise, if you want to apply transfer learning over our already trained network, or use it as is, we provide our already trained best model as a checkpoint (for retraining purposes) and as a frozen graph (.pb ready to deploy): https://zenodo.org/record/3544298#.Xc_IUMYh3CI

The steps to perform the network inference and obtain the detections of images from a frozen model are also provided in the Google object detection API repository. Although, we suggest swapping the evaluaate.py script for the one provided in this repository, as it will generate a txt file for each analysed image, indicating the bounding box, class, and confidence of the detected instances. This txt file will later be used to optimize the detections and generate de quantification.

# 2 - Optimize and evaluate the obtained detections.

Once the desired images have been forwarded to the network and it has generated the predictions, the user can post-process them by running the metrics/lib/nms.py script, deleting all detections that overlap with each other more than a desired percentage.

To evaluate a set of predictions, the user needs to execute the metrics/pascalvoc.py script. It will perform a confidence threshold and provide the AP, mAP, Recall and Precision metrics for the best found confidence threshold.

Finally, the user can delete all predictions above a certain threshold by using the metrcs/lib/cthr.py script and then draw the remaining predictions' bounding boxer over their corresponding images by executing the metrics/lib/printbb.py script.

# 3 - Quantification and monitoring algorithm

To apply the quantification and monitoring algorithm to a video sequence, the user first needs to forward the sequence, as images, into the trained network, as specified in step 1. Once the detections are obtained, the instance quantification can be obtained by running the quantification.py script
