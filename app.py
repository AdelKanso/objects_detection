import tkinter as tk

from utilities.camera_detection import show_objects_detection
from views.home_view import HomeView
from views.objects_detection_view import ObjectsDetectionView
from views.face_detection_view import FaceDetectionView

class App(tk.Tk):
	
	def __init__(self, *args, **kwargs): 
		tk.Tk.__init__(self, *args, **kwargs)
		self.attributes("-fullscreen", True)
		self.title("Object Detection")
		container = tk.Frame(self) 
		container.pack(side = "top", fill = "both", expand = True) 
		container.grid_rowconfigure(0, weight = 1)
		container.grid_columnconfigure(0, weight = 1)
		self.frames = {} 
		
		for F in (HomeView, FaceDetectionView,ObjectsDetectionView):
			frame = F(container, self)
			self.frames[F] = frame 
			frame.grid(row = 0, column = 0, sticky ="nsew")
		self.show_frame(HomeView)
	
	def show_frame(self, cont):
		frame = self.frames[cont]
		if cont is ObjectsDetectionView:
			show_objects_detection()
		else:
			frame.tkraise()

app = App()
app.mainloop()