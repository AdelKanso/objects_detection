import tkinter as tk

from views.objects_detection_view import ObjectsDetectionView
from views.face_detection_view import FaceDetectionView

class HomeView(tk.Frame):
	def __init__(self, parent, controller): 
		tk.Frame.__init__(self, parent)
		self.pack( fill='both', expand=True)

		self.label = tk.Label(self, text ="Detect      ", font=('Helvetica', 30))
		self.label.grid(row=0, column=1, columnspan=3,  sticky='nsew')  # Change to 'ew' for horizontal centering

		self.face_button = tk.Button(self, text='Face', font=('Helvetica', 20),command = lambda : controller.show_frame(FaceDetectionView), width=60)  # Set width only
		self.face_button.grid(row=1, column=1, pady=200, padx=100, sticky='nsew')

		self.general_button = tk.Button(self, text='Unknown!', font=('Helvetica', 20),command = lambda : controller.show_frame(ObjectsDetectionView), width=60)  # Set width only
		self.general_button.grid(row=1, column=3, pady=200, padx=100, sticky='nsew')

		self.grid_rowconfigure(0, weight=1)
		self.grid_rowconfigure(1, weight=1)
		self.grid_columnconfigure(0, weight=1)
		self.grid_columnconfigure(1, weight=1)
		self.grid_columnconfigure(2, weight=1)
		self.grid_columnconfigure(3, weight=1)
		self.grid_columnconfigure(4, weight=1)
