# Jellyfish object detection

This repository aims to provide the necesary tools and knowledge to operate our jellyfish object detection and monitoring algorithm. The following sections provide a step by step explaination on how to:

1 - Train an object detection neural network.

2 - Optimize the obtained detections.

3 - Apply our quantification and monitoring algorithm to a video sequence

# 1 - Train and infer an object detection neural network.

To train an object detection neural network from scratch, we reference to the Google object detection API, wich uses Tensorflow: https://github.com/tensorflow/models/tree/196d173a24613a045e641ef21ba9863c77bd1e2f/research/object_detection.

We provide our training data in the following link: https://zenodo.org/record/3537652#.XclltcYh3CI

Otherwise, if you want to aply transfer learning over an already trained networ on the provide dataset, or use it as is, we provide our already trained best model as a checkpoint (for retraining purposes) and as a frozen graph (.pb ready to deploy): https://zenodo.org/record/3544298#.Xc_IUMYh3CI

The steps to perform the nertwork inference and obtain the detections of images from a frozen model are also provided in the Google object detection API repository.

# 2 - Optimize and evaluate the obtained detections.

Once the desired images have been forwarded to the network and it has generated the predictions, they can post-processed by running the metrics/lib/nms.py script,all detections that overlapp with each other more than a desired % all deleted.

To perform the evaluation of a set of predictions, the user needs to execute the metrics/pascalvoc.py script. It will perform a confidence threshold and provide the AP, mAP, Recall and Precision metrics for the best found confidence threshold.

Finally, the user can delete all predictions above a certain chreshold by using the metrcs/lib/cthr.py script and then draw the remaining predicitons' bounding boxer over their corresponding images by executing the metrics/lib/printbb.py script.

# 3 - Apply our quantification and monitoring algorithm to a video sequence 





