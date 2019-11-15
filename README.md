# Jellyfish object detection

This repository aims to provide the necesary tools and knowledge to operate our jellyfish object detection and monitoring algorithm. The following sections provide a step by step explaination on how to:

1 - Train an object detection neural network.

2 - Use an already trained model to infer images and obtain detections.

3 - Optimize the obtained detections.

4 - Apply our quantification and monitoring algorithm to a video sequence

# 1 - Train an object detection neural network.

To train an object detection neural network from scratch, we reference to the Google object detection API, wich uses Tensorflow: https://github.com/tensorflow/models/tree/master/research/object_detection. 

We provide our training data in the following link: https://zenodo.org/record/3537652#.XclltcYh3CI

Otherwise, if you want to aply transfer learning over an already trained networ on the provide dataset, or use it as is, we provide our already trained best model: https://holder.com


