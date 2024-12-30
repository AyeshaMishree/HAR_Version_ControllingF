import numpy as np
import argparse
import imutils
import sys
import cv2

# Function to parse command-line arguments
def parse_arguments():
    argv = argparse.ArgumentParser()
    argv.add_argument("-m", "--model", required=True, help="specify path to pre-trained model")
    argv.add_argument("-c", "--classes", required=True, help="specify path to class labels file")
    argv.add_argument("-i", "--input", type=str, default="", help="specify path to video file")
    argv.add_argument("-o", "--output", type=str, default="", help="path to output video file")
    argv.add_argument("-d", "--display", type=int, default=1, help="to display output frame or not")
    argv.add_argument("-g", "--gpu", type=int, default=0, help="whether or not it should use GPU")
    
    return argv.parse_args()  # Directly return the Namespace object

# Function to load class labels
def load_labels(labels_path):
    return open(labels_path).read().strip().split("\n")

# Function to load the pre-trained model
def load_model(model_path, use_gpu):
    print("Loading The Deep Learning Model For Human Activity Recognition")
    model = cv2.dnn.readNet(model_path)
    if use_gpu > 0:
        print("Setting preferable backend and target to CUDA...")
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return model

# Function to initialize the video stream
def initialize_video_stream(input_path):
    print("Accessing the video stream...")
    vs = cv2.VideoCapture(input_path if input_path else 0)
    fps = vs.get(cv2.CAP_PROP_FPS)
    print("Original FPS:", fps)
    return vs, fps

# Function to process frames and create a blob
def process_frames(video_stream, sample_duration, sample_size):
    frames = []  # frames for processing
    originals = []  # original frames

    for i in range(0, sample_duration):
        (grabbed, frame) = video_stream.read()
        if not grabbed:
            print("[INFO] No frame read from the stream - Exiting...")
            return None, None
        originals.append(frame)  # save original frame
        frame = imutils.resize(frame, width=400)
        frames.append(frame)

    # Create a blob from the frames
    blob = cv2.dnn.blobFromImages(frames, 1.0, (sample_size, sample_size),
                                  (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)
    
    return originals, blob

# Function to make predictions using the model
def make_predictions(model, blob, activity_labels):
    model.setInput(blob)
    outputs = model.forward()
    label = activity_labels[np.argmax(outputs)]
    return label

# Function to display the frame with the predicted label
def display_frame(frame, label):
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Function to save the output frame to a video file
def save_output_frame(writer, frame, output_path):
    if output_path != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    if writer is not None:
        writer.write(frame)

# Main function to run the activity recognition
def main():
    # Parse arguments
    args = parse_arguments()
    
    # Load labels, model, and video stream
    activity_labels = load_labels(args.classes)  # Corrected access
    model = load_model(args.model, args.gpu)
    video_stream, fps = initialize_video_stream(args.input)
    
    writer = None
    
    while True:
        originals, blob = process_frames(video_stream, 16, 112)  # Now passing as positional arguments
        
        if originals is None:
            video_stream.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        
        # Make prediction
        label = make_predictions(model, blob, activity_labels)
        
        # Process and display frames
        for frame in originals:
            display_frame(frame, label)
            
            if args.display > 0:
                cv2.imshow("Activity Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    video_stream.release()
                    if writer:
                        writer.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
            
            save_output_frame(writer, frame, args.output)
    
    # Release resources on exit
    video_stream.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
