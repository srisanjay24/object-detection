import cv2
import numpy as np

# Load the YOLOv3 model configuration and weights
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

# Load the classes file
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set the input and output layers
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()

# Load the image
img = cv2.imread("example.jpg")
height, width, channels = img.shape

# Preprocess the image for input to the neural network
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Set the input for the neural network
net.setInput(blob)

# Run the forward pass to detect objects
outs = net.forward(output_layers)

# Set confidence threshold and NMS threshold
conf_threshold = 0.5
nms_threshold = 0.4

class_ids = []
confidences = []
boxes = []

# Loop over each output layer
for out in outs:
    # Loop over each detection
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Find the top-left and bottom-right coordinates of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Apply non-maximum suppression to remove redundant overlapping boxes with lower confidences
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Iterate over the selected indices to draw the bounding boxes
if len(indices) > 0:
    for i in indices.flatten():
        box = boxes[i]
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, classes[class_ids[i]], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with detected objects
cv2.imshow("Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Successfully executed")
