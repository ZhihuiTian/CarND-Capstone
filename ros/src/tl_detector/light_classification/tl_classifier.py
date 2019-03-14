from styx_msgs.msg import TrafficLight
import tensorflow as tf
import os
import numpy as np
import rospy

LABELS = ['Green', 'Red', 'Yellow', 'Unknown']


class TLClassifier(object):
    def __init__(self, is_site):
        # TODO load classifier
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        if is_site:
            graph_file = curr_dir + '/real_model/frozen_inference_graph.pb'
        else:
            graph_file = curr_dir + '/sim_model/frozen_inference_graph.pb'

        self.detection_graph = self.load_graph(graph_file)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name(
            'detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name(
            'detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def filter_boxes(self, min_score, boxes, scores, classes):
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

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        with tf.Session(graph=self.detection_graph) as sess:
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                feed_dict={self.image_tensor: image_np})
            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)
            confidence_cutoff = 0.5
            boxes, scores, classes = self.filter_boxes(
                confidence_cutoff, boxes, scores, classes)
            for i in range(boxes.shape[0]):
                class_name = LABELS[classes[i] - 1]
                rospy.logdebug("TrafficLight is %s", class_name)
                if class_name == 'Red':
                    return TrafficLight.RED
                elif class_name == 'Green':
                    return TrafficLight.GREEN
                elif class_name == 'Yellow':
                    return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN
