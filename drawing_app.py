import os
import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
import requests
from io import BytesIO

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("My Drawing Canvas 28x28")

        self.canvas_size = 680
        self.image_size = 28
        self.brush_size = 20

        self.canvas = tk.Canvas(root, bg="black", width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        self.image = Image.new("L", (self.image_size, self.image_size), "black")
        self.draw = ImageDraw.Draw(self.image)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.predict_button = tk.Button(self.button_frame, text="Predict The Digit", command=self.predict_image)
        self.predict_button.pack(side="left")

        self.clear_button = tk.Button(self.button_frame, text="Erase", command=self.clear_canvas)
        self.clear_button.pack(side="right")

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)

        self.canvas.create_oval(x1, y1, x2, y2, fill="yellow", outline="yellow")

        scaled_x1, scaled_y1 = (x1 * self.image_size // self.canvas_size), (y1 * self.image_size // self.canvas_size)
        scaled_x2, scaled_y2 = (x2 * self.image_size // self.canvas_size), (y2 * self.image_size // self.canvas_size)
        self.draw.ellipse([scaled_x1, scaled_y1, scaled_x2, scaled_y2], fill="yellow")

    def predict_image(self):
        image_data = np.array(self.image).astype(np.uint8)
        image = Image.fromarray(image_data, mode='L')
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        try:
            response = requests.post("http://127.0.0.1:8000/predict", files={"file": ("image.png", img_byte_arr, "image/png")})
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json().get("predicted_digit")
            messagebox.showinfo("Result", f"Digit: {result}")
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Error", f"Failed to connect to the API: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.image_size, self.image_size), "black")
        self.draw = ImageDraw.Draw(self.image)

if __name__ == "__main__":
    root = tk.Tk()
    root.tk.call('tk', 'scaling', 4.0)
    app = DrawingApp(root)
    root.mainloop()

