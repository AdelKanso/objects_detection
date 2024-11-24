import tkinter as tk
from views.face_detection_view import FaceDetectionView


class HomeView(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.pack(fill="both", expand=True)
        self.configure(bg="#f0f8ff")  # Light blue background for the frame

        # Title Label
        self.label = tk.Label(
            self,
            text="Detection Options",
            font=("Helvetica", 30, "bold"),
            bg="#f0f8ff",
            fg="#333",
        )
        self.label.grid(row=0, column=0, columnspan=2, pady=20, sticky="n")

        # Create buttons
        self.create_button(
            text="Face Detection",
            command=lambda: controller.show_frame(FaceDetectionView),
            row=1,
            column=0,
        )
        self.create_button(
            text="Object Detection",
            command=lambda: controller.run_detection("Objects"),
            row=1,
            column=1,
        )
        self.create_button(
            text="Draw by Hand",
            command=lambda: controller.run_detection("Hand"),
            row=2,
            column=0,
        )
        self.create_button(
            text="Text Detection",
            command=lambda: controller.run_detection("Text"),
            row=2,
            column=1,
        )

        # Configure grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def create_button(self, text, command, row, column):
        """Helper function to create styled buttons."""
        button = tk.Button(
            self,
            text=text,
            font=("Helvetica", 18, "bold"),
            highlightbackground='#3E4149', 
            command=command,
            width=20,
            height=2,
            relief="raised",
            bd=1,
        )
        button.grid(row=row, column=column, padx=20, pady=20, sticky="nsew")
