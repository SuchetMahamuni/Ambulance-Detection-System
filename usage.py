import cv2
import time
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

threshold = 0.20
detection_model = tf.saved_model.load("tfod/training/exported-models/centernet_model/saved_model")      # Centernet MobileNet V2 FPN 640x640

category_index = label_map_util.create_categories_from_labelmap('tfod/training/annotations/label_map.pbtxt', use_display_name=True)
cam = cv2.VideoCapture(0)   # Input from device's webcam

category_index = {label['id']:label for label in category_index}

#cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    start = time.time()

    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    #input_frame = cv2.flip(cv2.resize(frame, (640, 640)), 1)
    input_frame = frame.copy()
    image_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

    input_tensor = tf.convert_to_tensor(image_rgb)[tf.newaxis, ...]

    with tf.device('/GPU:0'):
        detections = detection_model(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    viz_utils.visualize_boxes_and_labels_on_image_array(
        input_frame,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=9,
        min_score_thresh=threshold,
        agnostic_mode=False)
    
    # FPS counter
    end_time = time.time()
    fps = 1 / (end_time - start)
    cv2.putText(input_frame, f'FPS: {int(fps)}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display output
    cv2.imshow('Real-Time Object Detection', cv2.resize(input_frame, (800, 600)))

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()