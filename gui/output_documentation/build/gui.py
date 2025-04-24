from pathlib import Path
import tkinter as tk
from tkinter import Canvas, PhotoImage, Button, Scrollbar, Frame, Label
from PIL import Image, ImageTk
import os


class DocumentationPage(tk.Frame):
    OUTPUT_PATH = Path(__file__).parent
    ASSETS_PATH = OUTPUT_PATH / Path("assets/frame0")
    PDF_IMAGES_PATH = OUTPUT_PATH / Path("assets/docs_pages")  # Folder with JPGs

    def relative_to_assets(self, path: str) -> Path:
        return self.ASSETS_PATH / Path(path)

    def __init__(self, parent, switchback_callback):
        super().__init__(parent)
        self.switchback_callback = switchback_callback
        self.image_refs = []  # Store image references

        # Header Canvas setup
        canvas = Canvas(self, bg="#FFFFFF", bd=0, highlightthickness=0, relief="ridge", width=1440, height=75)
        canvas.pack(fill="x")

        canvas.create_rectangle(0.0, 0.0, 1440.0, 75.0, fill="#1069D0", outline="")

        self.image_image_1 = PhotoImage(file=self.relative_to_assets("image_1.png"))
        canvas.create_image(1320.0, 37.0, image=self.image_image_1)

        self.button_image_1 = PhotoImage(file=self.relative_to_assets("button_1.png"))
        button_1 = Button(
            self,
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.switchback_callback(),
            relief="flat"
        )
        button_1.place(x=1249.0, y=18.0, width=141.0, height=39.0)

        # Scrollable PDF content viewer
        self.display_pdf_images()

    def display_pdf_images(self):
        viewer_frame = Frame(self)
        viewer_frame.place(x=50, y=100, width=1340, height=850)

        canvas = Canvas(viewer_frame)
        scrollbar = Scrollbar(viewer_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Load JPGs
        for file in sorted(os.listdir(self.PDF_IMAGES_PATH)):
            if file.endswith((".jpg", ".jpeg", ".png")):
                img_path = self.PDF_IMAGES_PATH / file
                img = Image.open(img_path)
                img = img.resize((1200, int(img.height * (1200 / img.width))), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                self.image_refs.append(img_tk)  # Keep reference
                label = Label(scrollable_frame, image=img_tk)
                label.pack(pady=10)
