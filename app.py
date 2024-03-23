import cv2
import tkinter as tk
from tkinter import filedialog
from maze_solver import find_path
from maze_classifier import MazeClassifier
from PIL import Image, ImageTk


class MazeSolverGui:
    def __init__(self, window) -> None:
        self.window = window
        self.window.title("Maze Solver")
        self.window.geometry("900x750")
        self.uploaded_image = None
        self.panel_uploaded = None
        self.panel_result = None
        self.mazeClassifier = MazeClassifier()

        self.upload_btn = tk.Button(
            self.window,
            text="Upload Image",
            command=self.upload_image,
        )
        self.upload_btn.pack(pady=5)

        self.generate_btn = tk.Button(self.window, text="Generate Maze")  # command= )
        self.generate_btn.pack(pady=5)

        self.export_btn = tk.Button(
            self.window, text="Export", command=self.export_image, state="disabled"
        )
        self.export_btn.pack(side="bottom", pady=5)

        self.solve_btn = tk.Button(
            self.window, text="Solve", command=self.solve_maze, state="disabled"
        )
        self.solve_btn.pack(side="bottom", pady=5)

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename()
        return file_path

    def upload_image(self):

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
        image = cv2.resize(image, (250, 250))
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.panel_uploaded = tk.Label(window, image=image)
        self.panel_uploaded.image = image
        self.panel_uploaded.pack()

    def display_result_image(self, image):
        image = cv2.resize(image, (250, 250))
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.panel_result = tk.Label(window, image=image)
        self.panel_result.image = image
        self.panel_result.pack()

    def export_image(self, result_image):
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


if __name__ == "__main__":
    window = tk.Tk()
    gui = MazeSolverGui(window)
    window.mainloop()
