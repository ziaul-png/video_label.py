### based on the image_label code provided by tensorflow

import argparse
import cv2
import numpy as np
import tensorflow as tf
import PIL


tf.compat.v1.disable_eager_execution()

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.compat.v1.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(frame,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  image_reader =  frame
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.compat.v1.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.compat.v1.Session()
  return sess.run(normalized)


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
 

def load_labels(label_file):
  proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
  return [l.rstrip() for l in proto_as_ascii_lines]


if __name__ == "__main__":
  
  file_name = cv2.VideoCapture("blue box timer.mp4")
  model_file = \
    "inception_v3_2016_08_28_frozen.pb"
  label_file = "imagenet_slim_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"

  fps = int(file_name.get(cv2.CAP_PROP_FPS))

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)

  fourcc = cv2.VideoWriter_fourcc(*'XVID')  
  output_vid = cv2.VideoWriter('output.mp4',fourcc, fps, (299,299)) 

  while (file_name.isOpened()):
    (taken, frame)= file_name.read()
    if frame is None:
      break
    t = read_tensor_from_image_file(frame,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.compat.v1.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0], {
          input_operation.outputs[0]: t
      })
    results = np.squeeze(results)
    labels = load_labels(label_file)
    
    index = results.argmax()
    value = str(results[index])
    best_label = str(labels[index])
    
    np_im = np.array(tensor_to_image(t))
    text = value + " "+ best_label
    org = (6,280)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    thickness = 2
  
    if results[index] >= 0.80:
      cv2.putText(np_im, text, org, fontFace,  fontScale, (0, 255,0), thickness)
    else:
      cv2.putText(np_im , text, org, fontFace, fontScale, (0, 0, 255), thickness)
    
    output_vid.write(np_im)

  file_name.release()
  output_vid.release()
  cv2.destroyAllWindows()
