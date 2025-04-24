import cv2
from PIL import Image, ImageTk

class ImageDisplay:
    """
    ImageDisplay is a class for displaying and interacting with an image on a Tkinter canvas.
    It supports zooming and panning functionalities.
    Attributes:
        canvas (tkinter.Canvas): The canvas widget where the image is displayed.
        zoom_factor (float): The current zoom factor of the image.
        zoom_step (float): The step size for zooming in or out.
        original_image (numpy.ndarray): The original image to be displayed.
        tk_image (ImageTk.PhotoImage): The image converted for display on the Tkinter canvas.
        image_id (int): The ID of the image object on the canvas.
        offset_x (int): The horizontal offset for panning the image.
        offset_y (int): The vertical offset for panning the image.
        last_x (int): The last x-coordinate of the mouse during panning.
        last_y (int): The last y-coordinate of the mouse during panning.
    Methods:
        __init__(canvas, zoom_step=0.1):
            Initializes the ImageDisplay instance with a canvas and optional zoom step.
        set_image(image):
            Sets the image to be displayed on the canvas.
        calculate_default_zoom():
            Calculates the default zoom factor to fit the image within the canvas.
        zoom_image(event):
            Zooms the image in or out based on mouse wheel events.
        start_pan(event):
            Initializes the panning operation by recording the starting mouse position.
        pan_image(event):
            Pans the image based on mouse drag events.
        display_image():
            Displays the image on the canvas, applying zoom and pan transformations.
    """
    def __init__(self, canvas, zoom_step=0.1):
        self.canvas = canvas
        self.zoom_factor = 1.0
        self.zoom_step = zoom_step
        self.original_image = None
        self.tk_image = None
        self.image_id = None

        self.offset_x = 0
        self.offset_y = 0

        self.canvas.bind("<MouseWheel>", self.zoom_image)
        self.canvas.bind("<B1-Motion>", self.pan_image)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)

        self.last_x = 0
        self.last_y = 0

    def set_image(self, image):
        """Set the image to display."""
        self.original_image = image
        self.calculate_default_zoom()
        self.offset_x = 0
        self.offset_y = 0
        self.display_image()

    def calculate_default_zoom(self):
        """Calculate zoom factor to fit the image within the canvas."""
        if self.original_image is not None:
            self.canvas.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            img_height, img_width = self.original_image.shape[:2]

            # Calculate zoom factor to fit the image into the canvas
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            self.zoom_factor = min(scale_x, scale_y)

    def zoom_image(self, event):
        """Zoom the image in or out."""
        if event.delta > 0:  # Zoom in
            self.zoom_factor += self.zoom_step
        elif event.delta < 0:  # Zoom out
            self.zoom_factor = max(self.zoom_factor - self.zoom_step, 1.0)  # Prevent zooming out below original size
        self.display_image()

    def start_pan(self, event):
        """Initialize panning."""
        self.last_x = event.x
        self.last_y = event.y

    def pan_image(self, event):
        """Pan the image."""
        dx = event.x - self.last_x
        dy = event.y - self.last_y

        self.offset_x += dx
        self.offset_y += dy

        self.last_x = event.x
        self.last_y = event.y

        self.display_image()

    def display_image(self):
        """Display the image on the canvas."""
        if self.original_image is not None:
            height, width = self.original_image.shape[:2]

            # Resize image according to zoom factor
            resized_image = cv2.resize(
                self.original_image,
                (int(width * self.zoom_factor), int(height * self.zoom_factor)),
                interpolation=cv2.INTER_NEAREST,
            )

            # Convert to PIL image for Tkinter
            img_pil = Image.fromarray(resized_image)
            self.tk_image = ImageTk.PhotoImage(image=img_pil)

            # Clear previous image
            if self.image_id:
                self.canvas.delete(self.image_id)

            # Constrain offsets to keep the image within canvas bounds
            self.canvas.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            img_width = self.tk_image.width()
            img_height = self.tk_image.height()

            # Update offsets to prevent the image from being panned outside the canvas
            self.offset_x = min(max(self.offset_x, canvas_width - img_width), 0)
            self.offset_y = min(max(self.offset_y, canvas_height - img_height), 0)

            # Draw the image on the canvas
            x = self.offset_x
            y = self.offset_y

            self.image_id = self.canvas.create_image(x, y, anchor="nw", image=self.tk_image)