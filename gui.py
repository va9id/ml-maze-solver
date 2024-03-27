import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
from maze_solver import find_path
from maze_classifier import MazeClassifier
from PIL import Image, ImageTk
from maze_generator import generate_random_maze_image
from image_helper import resize_image

class CustomButton(tk.Button):
    def __init__(self, master=None, **kw):
        tk.Button.__init__(self, master, **kw)
        self.configure(
            width=15,
            borderwidth=0,
            bg="white",
            fg="black",
            font=("Helvetica", 12, "bold")
        )

class MazeSolverGui:
    def __init__(self, window) -> None:
        self.window = window
        self.window.resizable(False, False)
        self.window.configure(background="#1B1B1B")
        self.window.title("Maze Solver")
        self.window.geometry("650x425")
        self.current_image = None
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
            command=self.generate_maze,
        )  
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
        self.window.mainloop()

    def display_popup(self, title, message, is_error=True):
        '''
        Display popup dialog
        '''
        messagebox.showerror(title, message) if is_error else messagebox.showinfo(title, message)

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
        if image_path and not image_path.endswith((".jpg", ".jpeg", ".png")):
            self.display_popup(
                "Invalid file type", 
                "Please select an image file with one of the following extensions: .jpg, .jpeg, .png"
            )
            self.export_btn.config(state="disabled")
            self.solve_btn.config(state="disabled")
            return
        if image_path:
            self.export_btn.config(state="disabled")
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            self.display_image(image)
            self.current_image = image
            self.solve_btn.config(state="normal")
        else:
            self.display_popup(
                "No image selected", 
                "No image was selected, please retry!"
            )
            self.export_btn.config(state="disabled")
            self.solve_btn.config(state="disabled")
            return
        
    def generate_maze(self):
        '''
        Handles generate a maze when the "Generate Maze" button is clicked
        '''
        if self.panel_uploaded is not None:
            self.panel_uploaded.destroy()

        if self.panel_result is not None:
            self.panel_result.destroy()

        image = cv2.cvtColor(np.array(generate_random_maze_image(), dtype='uint8'), cv2.COLOR_GRAY2BGR)
        self.export_btn.config(state="disabled")
        self.display_image(image)
        self.current_image = image
        self.solve_btn.config(state="normal")

    def solve_maze(self):
        '''
        Handles sovling the maze when the "Solve" button is clicked
        '''
        if self.current_image is None:
            self.display_popup(
                "No image selected",
                "No image has been selected to solve the maze for!"
            )
            return
        try:
            if self.mazeClassifier.is_maze(self.current_image):
                result_image = find_path(self.current_image)

                if self.panel_result is not None:
                    self.panel_result.destroy()

                self.display_result_image(result_image)
                self.solve_btn.config(state="disabled")
                self.export_btn.config(state="normal")
                self.export_btn["command"] = lambda: self.export_image(result_image)
            else:
                self.display_popup(
                    "Non maze image",
                    "The image does not look like a maze!"
                )
                return
        except Exception as error:
            print(error)
            self.display_popup(
                "Solving Error",
                "The maze could not be solved!"
            )
            return

    def display_image(self, image):
        '''
        Displays the maze image onto the GUI
        '''
        image = resize_image(image)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.panel_uploaded = tk.Label(self.window, image=image)
        self.panel_uploaded.image = image
        self.panel_uploaded.grid(row=1, column=0, columnspan=2, pady=20)

    def display_result_image(self, image):
        '''
        Displays the solved maze image onto the GUI
        '''
        image = resize_image(image)
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
                self.display_popup(
                    "Image saved",
                    "Solved maze image saved successfully!",
                    False
                )
            except Exception:
                self.display_popup(
                    "Export sovled maze error",
                    "Failed to save image!"
                )
