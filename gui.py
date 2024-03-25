import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from maze_solver import find_path
from maze_classifier import MazeClassifier
from PIL import Image, ImageTk
from constants import IMAGE_SIZE

class CustomButton(tk.Button):
    def __init__(self, master=None, **kw):
        tk.Button.__init__(self, master, **kw)
        self.configure(
            width=15,
            borderwidth=0,
            bg="#007FFF",
            fg="white",
            font=("Helvetica", 12, "bold")
        )

class MazeSolverGui:
    def __init__(self, window) -> None:
        self.window = window
        self.window.resizable(False, False)
        self.window.configure(background="#1B1B1B")
        self.window.title("Maze Solver")
        self.window.geometry("650x425")
        self.uploaded_image = None
        self.panel_uploaded = None
        self.panel_result = None
        self.mazeClassifier = MazeClassifier()

        self.upload_btn = CustomButton(
            self.window,
            text="Upload Image",
            command=self.upload_image,
        )
        self.upload_btn.grid(row=0, column=0, pady=20, padx=20, sticky="n")

        self.generate_btn = CustomButton(
            self.window, 
            text="Generate Maze", 
        )  # command= )
        self.generate_btn.grid(row=0, column=1, pady=20, padx=20, sticky="n")

        self.solve_btn = CustomButton(
            self.window, 
            text="Solve", 
            command=self.solve_maze, 
            state="disabled", 
        )
        self.solve_btn.grid(row=0, column=2, pady=20, padx=20, sticky="n")

        self.export_btn = CustomButton(
            self.window, 
            text="Export", 
            command=self.export_image, 
            state="disabled", 
        )
        self.export_btn.grid(row=0, column=3, pady=20, padx=20, sticky="n")

        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure((0, 1, 2, 3), weight=1)

    def open_file_dialog(self):
        '''
        Opens the file dialog
        '''
        file_path = filedialog.askopenfilename()
        return file_path

    def upload_image(self):
        '''
        Handles uploading an image when the "Upload Image" button is clicked
        '''
        if self.panel_uploaded is not None:
            self.panel_uploaded.destroy()

        if self.panel_result is not None:
            self.panel_result.destroy()

        image_path = self.open_file_dialog()
        if image_path:
            self.export_btn.config(state="disabled")
            image = cv2.imread(image_path)
            self.display_uploaded_image(image)
            self.uploaded_image = image
            self.solve_btn.config(state="normal")
        else:
            print("No image selected, exiting!")
            return

    def solve_maze(self):
        '''
        Handles sovling the maze when the "Solve" button is clicked
        '''
        if self.uploaded_image is None:
            print("No image selected, can not solve!")
            return
        try:
            if self.mazeClassifier.is_maze(self.uploaded_image):
                result_image = find_path(self.uploaded_image)

                if self.panel_result is not None:
                    self.panel_result.destroy()

                self.display_result_image(result_image)
                self.solve_btn.config(state="disabled")
                self.export_btn.config(state="normal")
                self.export_btn["command"] = lambda: self.export_image(result_image)
            else:
                print("The uploaded image does not look like a maze!")
                return
        except Exception as error:
            print(error)
            print("The uploaded maze could not be solved!")
            return

    def display_uploaded_image(self, image):
        '''
        Displays the uploaded maze image onto the GUI
        '''
        image = cv2.resize(image, IMAGE_SIZE)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.panel_uploaded = tk.Label(self.window, image=image)
        self.panel_uploaded.image = image
        self.panel_uploaded.grid(row=1, column=0, columnspan=2, pady=20)

    def display_result_image(self, image):
        '''
        Displays the solved maze image onto the GUI
        '''
        image = cv2.resize(image, IMAGE_SIZE)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.panel_result = tk.Label(self.window, image=image)
        self.panel_result.image = image
        self.panel_result.grid(row=1, column=2, columnspan=2, pady=20) 

    def export_image(self, result_image):
        '''
        Handles allowing the user to export/save the sovled maze image
        '''
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile="solved_maze.png"
        )
        if file_path:
            try:
                cv2.imwrite(file_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                print("Image saved successfully")
            except Exception as error:
                print(error)
                print("Failed to save image")
