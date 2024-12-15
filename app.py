import tkinter as tk
from controllers.app_controller import AppController
from views.home_view import HomeView


class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attributes("-fullscreen", True)
        self.title("Object Detection")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Initialize the controller
        self.controller = AppController(self)

        # Add frames to the controller
        frame = HomeView(container, self.controller)
        self.controller.add_frame(HomeView, frame)
        frame.grid(row=0, column=0, sticky="nsew")

        # Show the initial frame
        self.controller.show_frame(HomeView)

if __name__ == "__main__":
    app = App()
    app.mainloop()
