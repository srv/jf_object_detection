# Jellyfish object detection

This repository aims to provide the necessary tools and knowledge to operate our jellyfish object detection and monitoring algorithm. The following sections provide a step by step explanation on how to:

1 - Train and infer an object detection neural network.

2 - Optimize and evaluate the obtained detections.

3 - Apply our quantification and monitoring algorithm to a video sequence

# 1 - Train and infer an object detection neural network.

To train an object detection neural network from scratch, we reference to the Google object detection API, wich uses Tensorflow: https://github.com/tensorflow/models/tree/196d173a24613a045e641ef21ba9863c77bd1e2f/research/object_detection.

We provide our training data in the following link: https://zenodo.org/record/3537652#.XclltcYh3CI

Otherwise, if you want to apply transfer learning over our already trained network, or use it as is, we provide our already trained best model as a checkpoint (for retraining purposes) and as a frozen graph (.pb ready to deploy): https://zenodo.org/record/3544298#.Xc_IUMYh3CI

The steps to perform the network inference and obtain the detections of images from a frozen model are also provided in the Google object detection API repository.

# 2 - Optimize and evaluate the obtained detections.

Once the desired images have been forwarded to the network and it has generated the predictions, the user can post-process them by running the metrics/lib/nms.py script, deleting all detections that overlap with each other more than a desired percentage.

To perform the evaluation of a set of predictions, the user needs to execute the metrics/pascalvoc.py script. It will perform a confidence threshold and provide the AP, mAP, Recall and Precision metrics for the best found confidence threshold.

Finally, the user can delete all predictions above a certain threshold by using the metrcs/lib/cthr.py script and then draw the remaining predictions' bounding boxer over their corresponding images by executing the metrics/lib/printbb.py script.

# 3 - Apply our quantification and monitoring algorithm to a video sequence 





