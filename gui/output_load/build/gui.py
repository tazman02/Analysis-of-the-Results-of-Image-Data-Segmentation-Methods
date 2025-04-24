from pathlib import Path
import tkinter as tk
from tkinter import Canvas, PhotoImage, Button
from PIL import Image, ImageTk  # To resize images dynamically



class LoadPage(tk.Frame):
    OUTPUT_PATH = Path(__file__).parent
    ASSETS_PATH = OUTPUT_PATH / Path("assets/frame0")


    def relative_to_assets(self, path: str) -> Path:
        return self.ASSETS_PATH / Path(path)

    def __init__(self, parent, switch_callback, load_datasets_callback, load_groundtruth_callback, set_class_names_callback):
        super().__init__(parent)

        # Store the callback for loading datasets
        self.load_datasets_callback = load_datasets_callback
        self.load_groundtruth_callback = load_groundtruth_callback
        self.set_class_names_callback = set_class_names_callback
        self.switch_callback = switch_callback


        # Canvas setup
        canvas = Canvas(self, bg="#FFFFFF", bd=0, highlightthickness=0, relief="ridge",width=1440, height=1000)
        canvas.pack(fill="both", expand=True)  # Make the canvas expand to fill the frame

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
            x=212.0,
            y=18.0,
            width=154.0,
            height=35.0
        )

        self.button_image_2 = PhotoImage(
            file=self.relative_to_assets("button_2.png"))
        button_2 = Button(
            self,
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.switch_callback("DocumentationPage"),
            relief="sunken",
            bd=0
        )
        button_2.place(
            x=1265.0,
            y=18.0,
            width=135.0,
            height=35.0
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
            width=60.0,
            height=35.0
        )

        self.button_image_4 = PhotoImage(
            file=self.relative_to_assets("button_4.png"))
        button_4 = Button(
            self,
            image=self.button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.switch_callback("MetricsPage"),
            relief="sunken",
            bd=0
        )
        button_4.place(
            x=366.0,
            y=18.0,
            width=100.0,
            height=35.0
        )

        canvas.create_rectangle(
            586.0,
            258.0,
            986.0,
            760.0,
            fill="#CCE3FD",
            outline="")

        canvas.create_rectangle(
            1067.0,
            258.0,
            1367.0,
            760.0,
            fill="#CCE3FD",
            outline="")

        canvas.create_rectangle(
            105.0,
            258.0,
            505.0,
            760.0,
            fill="#CCE3FD",
            outline="")

        self.button_image_5 = PhotoImage(
            file=self.relative_to_assets("button_5.png"))
        button_5 = Button(
            self,
            image=self.button_image_5,
            borderwidth=0,
            highlightthickness=0,
            command=self.set_class_names_callback,
            relief="flat",         # Flat style
            bg="#ffffff",          # Match the background color of the canvas
            activebackground="#ffffff"  # Prevent hover background color
        )
        button_5.place(
            x=1092.0,
            y=819.0,
            width=250.0,
            height=57.0
        )

        self.button_image_6 = PhotoImage(
            file=self.relative_to_assets("button_6.png"))
        button_6 = Button(
            self,
            image=self.button_image_6,
            borderwidth=0,
            highlightthickness=0,
            command=self.load_groundtruth_callback,
            relief="flat",         # Flat style
            bg="#ffffff",          # Match the background color of the canvas
            activebackground="#ffffff"  # Prevent hover background color
        )
        button_6.place(
            x=153.0,
            y=820.0,
            width=305.0,
            height=57.0
        )

        self.button_image_7 = PhotoImage(
            file=self.relative_to_assets("button_7.png"))
        button_7 = Button(
            self,
            image=self.button_image_7,
            borderwidth=0,
            highlightthickness=0,
            command=self.load_datasets_callback,
            relief="flat",         # Flat style
            bg="#ffffff",          # Match the background color of the canvas
            activebackground="#ffffff"  # Prevent hover background color
        )
        button_7.place(
            x=633.0,
            y=820.0,
            width=305.0,
            height=57.0
        )

        canvas.create_rectangle(
            0.0,
            0.0,
            1440.0,
            75.0,
            fill="#1069D0",
            outline="")

        self.image_image_1 = PhotoImage(
            file=self.relative_to_assets("image_1.png"))
        self.image_1 = canvas.create_image(
            300.0,
            36.0,
            image=self.image_image_1
        )

        self.image_image_2 = PhotoImage(
            file=self.relative_to_assets("image_2.png"))
        self.image_2 = canvas.create_image(
            1330.0,
            37.0,
            image=self.image_image_2
        )

