import tkinter as tk

import views.home_view

class FaceDetectionView(tk.Frame):
	
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		self.general_button = tk.Button(self, text='Back', font=('Helvetica', 20),command = lambda : controller.show_frame(views.home_view.HomeView), width=60)
		self.general_button.grid(row=0, column=1, columnspan=3,  sticky='nsew')

		self.grid_columnconfigure(0, weight=1)
		self.grid_columnconfigure(1, weight=1)
		self.grid_columnconfigure(2, weight=1)
		self.grid_columnconfigure(3, weight=1)
		self.grid_columnconfigure(4, weight=1)