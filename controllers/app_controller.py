import tkinter as tk
from utilities.object_detection import objects_detection
from utilities.text_detection import text_detection
from utilities.hand_detection import draw_by_hand

class AppController:
    def __init__(self, root):
        self.root = root
        self.frames = {}

    def add_frame(self, name, frame):
        self.frames[name] = frame

    def show_frame(self, name):
        """Raise the frame with the given name."""
        frame = self.frames.get(name)
        if frame:
            frame.tkraise()
        else:
            print(f"Frame '{name}' not found.")

    def run_detection(self, detection_type):
        """Perform detection based on the detection type."""
        if detection_type == "Objects":
            objects_detection()
        elif detection_type == "Hand":
            draw_by_hand()
        elif detection_type == "Text":
            text_detection()
        else:
            print(f"Unknown detection type: {detection_type}")
