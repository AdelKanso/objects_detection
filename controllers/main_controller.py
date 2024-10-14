from utilities.camera_detection import show_objects_detection
from views.face_detection_view import FaceDetectionView
from views.objects_detection_view import ObjectsDetectionView

class MainController:
    def __init__(self, main_view):
        self.main_view = main_view
        
        
        # Configure buttons to navigate to different views
        self.main_view.face_button.configure(command=self.show_face_detection)
        self.main_view.general_button.configure(command=self.show_objects_detection)

    def show_face_detection(self):
        self.face_detection_view = FaceDetectionView(self.main_view.master, self.main_view)
        self.main_view.frame.pack_forget()
        self.face_detection_view.pack(fill='both', expand=True)

    def show_objects_detection(self):
        # Initialize the face and objects detection views
        print("akkasokooko")


        self.objects_detection_view = ObjectsDetectionView(self.main_view.master, self.main_view)
        # Check if the objects detection view needs to be reset
        self.main_view.frame.pack_forget()  # Hide the main frame
        if not self.objects_detection_view.winfo_ismapped():  # Only pack if it's not already visible
            self.objects_detection_view.pack(fill='both', expand=True)  # Show objects detection view
