import cv2
import tkinter as tk
from tkinter import filedialog
from maze_solver import find_path
from maze_classifier import MazeClassifier
from PIL import Image, ImageTk

window = tk.Tk()
uploaded_image = None
panel_uploaded = None
panel_result = None


def open_file_dialog():
    file_path = filedialog.askopenfilename()
    return file_path


def upload_image():
    global uploaded_image
    global panel_uploaded
    global panel_result

    if panel_uploaded is not None:
        panel_uploaded.destroy()

    if panel_result is not None:
        panel_result.destroy()

    image_path = open_file_dialog()
    if image_path:
        export_btn.config(state="disabled")
        image = cv2.imread(image_path)
        display_uploaded_image(image)
        uploaded_image = image
        solve_btn.config(state="normal")
    else:
        print("No image selected, exiting!")
        return


def solve_maze():
    global uploaded_image
    global panel_result

    if uploaded_image is None:
        print("No image selected, can not solve!")
        return
    mazeClassifier = MazeClassifier()
    try:
        if mazeClassifier.is_maze(uploaded_image):
            result_image = find_path(uploaded_image)

            if panel_result is not None:
                panel_result.destroy()

            display_result_image(result_image)
            solve_btn.config(state="disabled")
            export_btn.config(state="normal")
            export_btn["command"] = lambda: export_image(result_image)
        else:
            print("The uploaded image does not look like a maze!")
            return
    except Exception as error:
        print(error)
        print("The uploaded maze could not be solved!")
        return


def display_uploaded_image(image):
    global panel_uploaded
    image = cv2.resize(image, (250, 250))
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    panel_uploaded = tk.Label(window, image=image)
    panel_uploaded.image = image
    panel_uploaded.pack()


def display_result_image(image):
    global panel_result
    image = cv2.resize(image, (250, 250))
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    panel_result = tk.Label(window, image=image)
    panel_result.image = image
    panel_result.pack()


def export_image(result_image):
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
    # main()
    window.title("Maze Solver")
    window.geometry("900x750")
    upload_btn = tk.Button(window, text="Upload Image", command=upload_image)
    upload_btn.pack(pady=5)

    generate_btn = tk.Button(window, text="Generate Maze")  # command= )
    generate_btn.pack(pady=5)

    export_btn = tk.Button(
        window, text="Export", command=export_image, state="disabled"
    )
    export_btn.pack(side="bottom", pady=5)

    solve_btn = tk.Button(window, text="Solve", command=solve_maze, state="disabled")
    solve_btn.pack(side="bottom", pady=5)

    window.mainloop()
