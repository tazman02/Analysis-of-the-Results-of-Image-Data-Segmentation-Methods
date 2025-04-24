from pathlib import Path
import tkinter as tk
from tkinter import Canvas, PhotoImage, Button
from PIL import Image, ImageTk  # To resize images dynamically



class AnalysisPage(tk.Frame):
    OUTPUT_PATH = Path(__file__).parent
    ASSETS_PATH = OUTPUT_PATH / Path("assets/frame0")

    def relative_to_assets(self, path: str) -> Path:
        return self.ASSETS_PATH / Path(path)

    def __init__(self, parent, switch_callback, next_image_callback, prev_image_callback):
        super().__init__(parent)
        
        self.switch_callback = switch_callback
        self.next_image_callback = next_image_callback
        self.prev_image_callback = prev_image_callback

        # Canvas setup
        canvas = Canvas(self, bg="#FFFFFF", bd=0, highlightthickness=0, relief="ridge",width=1440, height=1000)
        canvas.pack(fill="both", expand=True)  # Make the canvas expand to fill the frame

        # Load and resize the image
        image_path = self.relative_to_assets("image_1.png")
        image = Image.open(image_path)
        resized_image = image.resize((1440, 1024))  # Match canvas size
        self.image_image_1 = ImageTk.PhotoImage(resized_image)

        # Place the resized image on the canvas
        canvas.create_image(0, 0, image=self.image_image_1, anchor="nw")

        # Add rectangles
        canvas.create_rectangle(
            1002.0, 543.0, 1346.0, 583.0,
            fill="#FFFFFF", outline=""
        )
        canvas.create_rectangle(
            1002.0, 609.0, 1346.0, 649.0,
            fill="#FFFFFF", outline=""
        )

        # Add buttons
        self.button_image_1 = PhotoImage(file=self.relative_to_assets("button_1.png"))
        button_1 = Button(
            self,
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.prev_image_callback(),
            relief="flat"
        )
        button_1.place(x=1000.0, y=689.0, width=82.0, height=48.0)

        self.button_image_2 = PhotoImage(file=self.relative_to_assets("button_2.png"))
        button_2 = Button(
            self,
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.next_image_callback(),
            relief="flat"
        )
        button_2.place(x=1200.0, y=689.0, width=84.0, height=48.0)

        # Add text
        canvas.create_text(
            656.0, 83.0,
            anchor="nw",
            text="Metrics for image:",
            fill="#000000",
            font=("Inter", 16 * -1)
        )
        canvas.create_text(
            656.0, 500.0,
            anchor="nw",
            text="Select pats:",
            fill="#000000",
            font=("Inter", 16 * -1)
        )
        canvas.create_text(
            15.0, 83.0,
            anchor="nw",
            text="Preview:",
            fill="#000000",
            font=("Inter", 16 * -1)
        )
        canvas.create_text(
            15.0, 760.0,
            anchor="nw",
            text="Statistics for dataset:",
            fill="#000000",
            font=("Inter", 16 * -1)
        )
        canvas.create_text(
            940.0, 620.0,
            anchor="nw",
            text="Image ID:",
            fill="#52525B",
            font=("Inter", 12 * -1)
        )
        canvas.create_text(
            963.0, 555.0,
            anchor="nw",
            text="Class",
            fill="#52525B",
            font=("Inter", 12 * -1)
        )

        # More buttons
        self.button_image_3 = PhotoImage(file=self.relative_to_assets("button_7.png"))
        button_3 = Button(
            self,
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.switch_callback("LoadPage"),
            # command=lambda: print("button_3 clicked"),
            relief="sunken",
            bd=0
        )
        button_3.place(x=50.0, y=18.0, width=146.0, height=37.0)

        self.button_image_4 = PhotoImage(file=self.relative_to_assets("button_4.png"))
        button_4 = Button(
            self,
            image=self.button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.switch_callback("DocumentationPage"),
            relief="sunken",
            bd=0
        )
        button_4.place(x=1262.0, y=18.0, width=136.0, height=37.0)

        self.button_image_5 = PhotoImage(file=self.relative_to_assets("button_5.png"))
        button_5 = Button(
            self,
            image=self.button_image_5,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.switch_callback("ExportPage"),
            relief="sunken",
            bd=0
        )
        button_5.place(x=486.0, y=18.0, width=62.0, height=37.0)

        self.button_image_6 = PhotoImage(file=self.relative_to_assets("button_6.png"))
        button_6 = Button(
            self,
            image=self.button_image_6,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.switch_callback("MetricsPage"),
            relief="sunken",
            bd=0
        )
        button_6.place(x=369.0, y=18.0, width=101.0, height=37.0)
