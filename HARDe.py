# -----------------------------
#   USAGE
# -----------------------------
# python human_activity_recognition_deque.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt --input videos/example_activities.mp4
# python human_activity_recognition_deque.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

def parse_arguments():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Path to trained human activity recognition model")
    parser.add_argument("-c", "--classes", required=True, help="Path to class labels file")
    parser.add_argument("-i", "--input", type=str, default="", help="Optional path to video file")
    return vars(parser.parse_args())

def load_class_labels(path):
    """Load and return class labels from the specified file."""
    with open(path, "r") as file:
        return file.read().strip().split("\n")

def initialize_video_stream(input_path):
    """Initialize and return the video stream."""
    print("[INFO] Accessing the video stream...")
    return cv2.VideoCapture(input_path if input_path else 0)

def main():
    args = parse_arguments()

    # Load class labels
    CLASSES = load_class_labels(args["classes"])

    # Define sample parameters
    SAMPLE_DURATION = 16
    SAMPLE_SIZE = 112

    # Initialize frame queue
    frames = deque(maxlen=SAMPLE_DURATION)

    # Load the human activity recognition model
    print("[INFO] Loading the human activity recognition model...")
    net = cv2.dnn.readNet(args["model"])

    # Initialize video stream
    vs = initialize_video_stream(args["input"])

    while True:
        # Read the frame from the video stream
        grabbed, frame = vs.read()

        if not grabbed:
            print("[INFO] No frame read from the video stream - Exiting...")
            break

        # Resize frame and add to queue
        frame = imutils.resize(frame, width=400)
        frames.append(frame)

        # Wait until the queue is filled
        if len(frames) < SAMPLE_DURATION:
            continue

        # Construct blob from frames
        blob = cv2.dnn.blobFromImages(
            frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE),
            (114.7748, 107.7354, 99.4750), swapRB=True, crop=True
        )
        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)

        # Perform prediction
        net.setInput(blob)
        outputs = net.forward()
        label = CLASSES[np.argmax(outputs)]

        # Display prediction on frame
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show frame
        cv2.imshow("Activity Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

