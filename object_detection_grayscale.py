import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os

# Frozen inference graph files. NOTE: change the path to where you saved the models.
SSD_GRAPH_FILE = './model/faster_rcnn_model.pb'


def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)
    
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].
    
    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width
    
    return box_coords

def draw_boxes(image, boxes, classes, thickness=10):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('./resources/raleway.ttf', 40)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        if class_id == 1:
            draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill='black')
        
def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

detection_graph = load_graph(SSD_GRAPH_FILE)

# The input placeholder for the image.
# `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

# The classification of the object (integer id).
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

def run_detection():
    images = []
    image_objects = []
    names = []
    for img_name in os.listdir('./images'):
        # Load an image
        image_path = './images/' + img_name
        image = Image.open(image_path).convert('L')
        image_objects.append(image)
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        image_np = image_np[..., np.newaxis]
        image_np = tf.image.grayscale_to_rgb(tf.convert_to_tensor(image_np))
        image_np = np.asarray(image_np, dtype=np.uint8)
        images.append(image_np)
        names.append(img_name)

    with tf.compat.v1.Session(graph=detection_graph) as sess:
        for i in range(len(images)):
            # Actual detection.
            (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes], 
                                                feed_dict={image_tensor: images[i]})
            
            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            confidence_cutoff = 0.8
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            width, height = image_objects[i].size
            box_coords = to_image_coords(boxes, height, width)

            # Each class with be represented by a differently colored box
            draw_boxes(image_objects[i], box_coords, classes)

            image_objects[i].save('./out/out_' + names[i])

run_detection()
