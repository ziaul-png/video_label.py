# video_label.py
# Video-labelling-computer-vision-

check out: tensorflow/examples/label_image/README.md
check out: [tensorflow/examples/label_image/README.md](https://github.com/tensorflow/tensorflow/blob/5a6fc06bf80f84587490f61c27a03b7ee5457563/tensorflow/examples/label_image/README.md)

This project is based on label_image provided by Tensorflow: tensorflow/tensorflow/examples/label_image/

Instead of an image, It takes a video as input, reads it frame by frame and, for each frame, classifies it, and writes a colored text of the label with the highest confidence value (in green if its >= 80%, else in red).

This project takes a video as input, reads it frame by frame and, for each frame, classifies it, and shows athe result on the image itself by displaying the label with the highest confidence value (in green if its >= 80%, else in red.
