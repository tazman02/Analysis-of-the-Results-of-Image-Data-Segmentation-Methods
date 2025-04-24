from pathlib import Path
import tkinter as tk
from tkinter import Canvas, PhotoImage, Button
from PIL import Image, ImageTk  # To resize images dynamically



class ExportPage(tk.Frame):
    OUTPUT_PATH = Path(__file__).parent
    ASSETS_PATH = OUTPUT_PATH / Path("assets/frame0")

    def relative_to_assets(self, path: str) -> Path:
        return self.ASSETS_PATH / Path(path)

    def __init__(self, parent, switch_callback, export_callback):
        super().__init__(parent)
        
        self.switch_callback = switch_callback
        self.export_callback = export_callback

        # Canvas setup
        canvas = Canvas(self, bg="#FFFFFF", bd=0, highlightthickness=0, relief="ridge",width=1440, height=1000)
        canvas.pack(fill="both", expand=True)  # Make the canvas expand to fill the frame

        canvas.create_rectangle(
            0.0,
            0.0,
            1440.0,
            75.0,
            fill="#1069D0",
            outline="")

        canvas.create_rectangle(
            15.0,
            136.0,
            645.0,
            990.0,
            fill="#CCE3FD",
            outline="")

        canvas.create_text(
            15.0,
            104.0,
            anchor="nw",
            text="Preview:",
            fill="#000000",
            font=("Inter", 16 * -1)
        )

        self.image_image_1 = PhotoImage(
            file=self.relative_to_assets("image_1.png"))
        self.image_1 = canvas.create_image(
            1174.0,
            156.0,
            image=self.image_image_1
        )

        canvas.create_text(
            940.0,
            234.0,
            anchor="nw",
            text="Dest path",
            fill="#52525B",
            font=("Inter", 12 * -1)
        )

        canvas.create_text(
            950.0,
            145.0,
            anchor="nw",
            text="Format:",
            fill="#52525B",
            font=("Inter", 12 * -1)
        )

        self.button_image_1 = PhotoImage(
            file=self.relative_to_assets("button_1.png"))
        button_1 = Button(
            self,
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.export_callback(),
            relief="flat",         # Flat style
            bg="#ffffff",          # Match the background color of the canvas
            activebackground="#ffffff"  # Prevent hover background color
        )
        button_1.place(
            x=1002.0,
            y=224.0,
            width=344.0,
            height=40.0
        )

        self.button_image_2 = PhotoImage(
            file=self.relative_to_assets("button_2.png"))
        button_2 = Button(
            self,
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.export_callback(),
            relief="flat"
        )
        button_2.place(
            x=1103.0,
            y=297.0,
            width=98.0,
            height=48.0
        )

        self.image_image_2 = PhotoImage(
            file=self.relative_to_assets("image_2.png"))
        self.image_2 = canvas.create_image(
            300.0,
            36.0,
            image=self.image_image_2
        )

        self.image_image_3 = PhotoImage(
            file=self.relative_to_assets("image_3.png"))
        self.image_3 = canvas.create_image(
            728.0,
            288.0,
            image=self.image_image_3
        )

        self.image_image_4 = PhotoImage(
            file=self.relative_to_assets("image_4.png"))
        self.image_4 = canvas.create_image(
            1330.0,
            37.0,
            image=self.image_image_4
        )

        self.button_image_3 = PhotoImage(
            file=self.relative_to_assets("button_3.png"))
        button_3 = Button(
            self,
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.switch_callback("AnalysisPage"),
            relief="sunken",
            bd=0
        )
        button_3.place(
            x=197.0,
            y=17.0,
            width=148.0,
            height=37.0
        )

        self.button_image_4 = PhotoImage(
            file=self.relative_to_assets("button_4.png"))
        button_4 = Button(
            self,
            image=self.button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.switch_callback("DocumentationPage"),
            relief="sunken",
            bd=0
        )
        button_4.place(
            x=1261.0,
            y=17.0,
            width=136.0,
            height=37.0
        )

        self.button_image_5 = PhotoImage(
            file=self.relative_to_assets("button_5.png"))
        button_5 = Button(
            self,
            image=self.button_image_5,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.switch_callback("LoadPage"),
            relief="sunken",
            bd=0
        )
        button_5.place(
            x=48.0,
            y=17.0,
            width=138.0,
            height=37.0
        )

        self.button_image_6 = PhotoImage(
            file=self.relative_to_assets("button_6.png"))
        button_6 = Button(
            self,
            image=self.button_image_6,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.switch_callback("MetricsPage"),
            relief="sunken",
            bd=0
        )
        button_6.place(
            x=362.0,
            y=17.0,
            width=101.0,
            height=37.0
        )
