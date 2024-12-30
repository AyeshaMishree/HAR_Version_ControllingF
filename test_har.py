import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import cv2
import numpy as np
import os

from HAR import parse_arguments, load_labels, load_model, process_frames
from HAR import display_frame
from HAR import make_predictions
from HAR import save_output_frame


class TestHumanActivityRecognition(unittest.TestCase):

    # Test parse_arguments function
    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_arguments(self, mock_parse_args):
        mock_parse_args.return_value = {
            'model': 'model.pb',
            'classes': 'classes.txt',
            'input': 'video.mp4',
            'output': 'output.mp4',
            'display': 1,
            'gpu': 0
        }
        args = parse_arguments()
        self.assertEqual(args['model'], 'model.pb')
        self.assertEqual(args['classes'], 'classes.txt')
        self.assertEqual(args['input'], 'video.mp4')
    
    # Test load_labels function
    def test_load_labels(self):
        with patch('builtins.open', mock_open(read_data="Walking\nRunning\nJumping")):
            labels = load_labels("classes.txt")
            self.assertEqual(labels, ["Walking", "Running", "Jumping"])
    
    # Test load_model function with a mock model
    @patch('cv2.dnn.readNet')
    def test_load_model(self, mock_readNet):
        mock_model = MagicMock()
        mock_readNet.return_value = mock_model
        model = load_model('model.pb', use_gpu=False)
        self.assertEqual(model, mock_model)
    
    # Test process_frames function (mocking cv2.VideoCapture and imutils.resize)
    @patch('cv2.VideoCapture')
    @patch('imutils.resize')
    def test_process_frames(self, mock_resize, mock_video_capture):
        mock_capture = MagicMock()
        mock_video_capture.return_value = mock_capture
        mock_capture.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))  # Mock frame
        mock_resize.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        originals, blob = process_frames(mock_capture, 16, 112)
        self.assertIsNotNone(originals)
        self.assertIsNotNone(blob)
        self.assertEqual(len(originals), 16)
        mock_capture.read.assert_called()
        mock_resize.assert_called()
    
    # Test make_predictions function
    @patch('cv2.dnn.readNet')
    def test_make_predictions(self, mock_readNet):
        mock_model = MagicMock()
        mock_readNet.return_value = mock_model
        mock_model.forward.return_value = np.array([[0.1, 0.7, 0.2]])  # Mock output (index 1 is highest)
        
        activity_labels = ['Walking', 'Running', 'Jumping']
        label = make_predictions(mock_model, np.zeros((1, 16, 112, 112, 3)), activity_labels)
        self.assertEqual(label, 'Running')
    
    # Test display_frame function
    @patch('cv2.putText')
    @patch('cv2.rectangle')
    def test_display_frame(self, mock_rectangle, mock_putText):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        label = "Running"
        
        display_frame(frame, label)
        
        mock_rectangle.assert_called()
        mock_putText.assert_called()
    
    # Test save_output_frame function
    @patch('cv2.VideoWriter')
    def test_save_output_frame(self, mock_video_writer):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_writer = MagicMock()
        mock_video_writer.return_value = mock_writer
        
        save_output_frame(mock_writer, frame, "output.mp4")
        mock_writer.write.assert_called_with(frame)
    
    # Test save_output_frame when no writer is initialized
    @patch('cv2.VideoWriter')
    def test_save_output_frame_no_writer(self, mock_video_writer):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        save_output_frame(None, frame, "output.mp4")
        mock_video_writer.assert_called()


# Custom test runner to print results
class CustomTestRunner(unittest.TextTestRunner):
    def run(self, test):
        result = super().run(test)
        print("\nTest Results:")
        for test_case in result.testsRun:
            outcome = "passed" if test_case not in result.failures and test_case not in result.errors else "failed"
            print(f"{test_case} : {outcome}")
        return result


if __name__ == '__main__':
    unittest.main(testRunner=CustomTestRunner())
