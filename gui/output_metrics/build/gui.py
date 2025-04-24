from pathlib import Path
import tkinter as tk
from tkinter import Canvas, PhotoImage, Button
from PIL import Image, ImageTk  # To resize images dynamically



class MetricsPage(tk.Frame):
    OUTPUT_PATH = Path(__file__).parent
    ASSETS_PATH = OUTPUT_PATH / Path("assets/frame0")

    def relative_to_assets(self, path: str) -> Path:
        return self.ASSETS_PATH / Path(path)

    def __init__(self, parent, switch_callback):
        super().__init__(parent)
        
        self.switch_callback = switch_callback

        # Canvas setup
        canvas = Canvas(self, bg="#FFFFFF", bd=0, highlightthickness=0, relief="ridge",width=1440, height=1000)
        canvas.pack(fill="both", expand=True)  # Make the canvas expand to fill the frame

        canvas.create_rectangle(
            0.0,
            2.0,
            1440.0,
            77.0,
            fill="#1069D0",
            outline="")

        image_path = self.relative_to_assets("image_1.png")
        image = Image.open(image_path)
        resized_image = image.resize((1440, 1024))  # Match canvas size
        self.image_image_1 = ImageTk.PhotoImage(resized_image)

        # Place the resized image on the canvas
        canvas.create_image(0, 0, image=self.image_image_1, anchor="nw")

        self.button_image_1 = PhotoImage(
            file=self.relative_to_assets("button_1.png"))
        button_1 = Button(
            self,
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.switch_callback("AnalysisPage"),
            relief="sunken",
            bd=0
        )
        button_1.place(
            x=195.0,
            y=18.0,
            width=160.0,
            height=37.0
        )

        self.button_image_2 = PhotoImage(
            file=self.relative_to_assets("button_2.png"))
        self.button_2 = Button(
            self,
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.switch_callback("DocumentatioPage"),
            relief="sunken",
            bd=0
        )
        self.button_2.place(
            x=1265.0,
            y=18.0,
            width=136.0,
            height=37.0
        )

        self.button_image_3 = PhotoImage(
            file=self.relative_to_assets("button_3.png"))
        button_3 = Button(
            self,
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.switch_callback("ExportPage"),
            relief="sunken",
            bd=0
        )
        button_3.place(
            x=483.0,
            y=18.0,
            width=62.0,
            height=37.0
        )

        self.button_image_4 = PhotoImage(
            file=self.relative_to_assets("button_4.png"))
        button_4 = Button(
            self,
            image=self.button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.switch_callback("LoadPage"),
            relief="sunken",
            bd=0
        )
        button_4.place(
            x=47.0,
            y=17.0,
            width=138.0,
            height=37.0
        )
