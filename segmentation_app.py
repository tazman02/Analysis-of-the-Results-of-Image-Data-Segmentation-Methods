"""
====================================================================================
Segmentation Visualization and Evaluation GUI Tool
====================================================================================

Author: Matus Guga
Email: matus.guga@vgd.sk
File: main.py
Date: April 2025

Description:
------------
This script is the main entry point for a graphical user interface (GUI) application
used for evaluating image segmentation results. It allows users to load predicted
segmentation masks and corresponding ground truth images, compute various evaluation
metrics (Dice, IoU, Accuracy, etc.), and visualize class-wise performance.

Main Features:
--------------
- Load and preview mask and ground truth image datasets
- Per-class and dataset-wide segmentation metric calculations
- Interactive visualization of misclassified areas
- Export functionality for metrics in CSV or JSON format
- GUI built with Tkinter and extended with themed widgets (ttk)
- Modular design separating GUI pages and logic components

Dependencies:
-------------
- OpenCV (cv2)
- NumPy
- SciPy
- Pillow (PIL)
- Tkinter (standard library)
- Custom modules: gui.*, lib.*

This application is designed to be user-friendly and extensible for research or
production evaluation of segmentation tasks in computer vision.

====================================================================================
"""

# -------------------- Imports --------------------
import os
import cv2
print(cv2.__version__)
import numpy as np
print(np.__version__)
import tkinter as tk
import threading
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import PhotoImage
#import torchmetrics as tm
import scipy
print(scipy.__version__)


from gui.output_analyse.build.gui import AnalysisPage
from gui.output_export.build.gui import ExportPage
from gui.output_metrics.build.gui import MetricsPage
from gui.output_documentation.build.gui import DocumentationPage
from gui.output_load.build.gui import LoadPage
from lib.class_manager import ClassManager
from lib.export_metrics import ExportMetrics
from lib.directory_loader import DirectoryLoader
from lib.image_display import ImageDisplay
from lib.legend import Legend
from lib.image_analyzer import ImageAnalyzer
from lib.metrics_calculator import MetricsCalculator

class SegmentationApp:
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.root.title("Segmentation Visualization")
        self.root.geometry("1440x1000")
        self.root.resizable(False, False)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))

        icon_path = os.path.join(script_dir, 'ico.png')

        icon = PhotoImage(file=icon_path)
        
        root.iconphoto(True, icon)
        
        self.current_idx = 0
        self.mask_images = []
        self.gt_images = []

        self.checkbox_states = {
            "Class": tk.BooleanVar(value=True),
            "Dice": tk.BooleanVar(value=True),
            "IoU": tk.BooleanVar(value=True),
            "Accuracy": tk.BooleanVar(value=True),
            "Precision": tk.BooleanVar(value=True),
            "Recall": tk.BooleanVar(value=True),
            "Specificity": tk.BooleanVar(value=True),
            "Fallout": tk.BooleanVar(value=True),
            "Fnr": tk.BooleanVar(value=True),
            "Volumetric similarity": tk.BooleanVar(value=True),
            "AUC": tk.BooleanVar(value=True),
            "GCE score": tk.BooleanVar(value=True),
            "Kappa": tk.BooleanVar(value=True),
            "AHD": tk.BooleanVar(value=True),
            "ASSD": tk.BooleanVar(value=True),
            "DSC": tk.BooleanVar(value=True),
            "Boundary IoU": tk.BooleanVar(value=True),
            "AP": tk.BooleanVar(value=True)
        }
        
        self.image_selection_input = None  # input for image selection export
        
        self.analysis_done = False
        
        # Configure root layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Main container frame
        self.container = tk.Frame(self.root, bg="white")
        self.container.grid(row=0, column=0, sticky="nsew")
        
        self.canvas =  self.create_canvas()
        
        self.canvas_load_mask = tk.Canvas(root, bg="blue", width=400, height=300)
        self.canvas_load_mask.image_refs = []  # Store image references
        
        
        self.canvas_load_mask.place_forget()
        self.canvas.place_forget()
        
        style = ttk.Style()
        style.configure("White.TFrame", background="white", relief="flat", borderwidth=0)
        
                # Legend
        self.legend_frame = tk.Frame(self.container, bg="white", highlightthickness=0)
        self.legend_frame.place(relx=0.5, rely=0.6, width=400, height=400)

        # Create Legend inside the frame
        self.legend = Legend(self.legend_frame, self.update_class_visualization,)
        
        self.legend_frame.place_forget()
        
                # Metrics table
        self.metrics_frame = ttk.Frame(self.container, relief="groove", padding=10)
        self.metrics_frame.place(relx=0.7, rely=0.55, relwidth=0.25, relheight=0.4)
        
        self.class_manager = ClassManager(self)
        
        
        self.analyzer = ImageAnalyzer(self.legend, self.class_manager)
        
        self.current_image_metrics_frame = ttk.Frame(self.container, relief="groove", padding=10)
        self.current_image_metrics_frame.place(relx=0.7, rely=0.2, relwidth=0.25, relheight=0.3)  # Adjust as needed
        self.dataset_metrics_frame = ttk.Frame(self.container, relief="groove", padding=10)
        self.dataset_metrics_frame.place(relx=0.7, rely=0.5, relwidth=0.25, relheight=0.3) 
        
        
        self.compare_button = ttk.Button(self.root, text="Compare Images", command=self.open_comparison_window)
        

        self.metrics_calculator = MetricsCalculator(self.metrics_frame, self.class_manager, self.analyzer, self, self.dataset_metrics_frame)

        self.current_image_metrics_calculator = MetricsCalculator(self.current_image_metrics_frame, self.class_manager, self.analyzer, self, self.dataset_metrics_frame)
        
        self.class_manager.set_metrics_calculator(self.metrics_calculator)

        # Hide this metrics frame initially
        self.current_image_metrics_frame.place_forget()
        self.dataset_metrics_frame.place_forget()
        
        self.metrics_frame.place_forget()
        

        # Initialize other components
        self.image_display = ImageDisplay(self.canvas)

        # Initialize LoadPage and pass the dataset loading method
        self.load_page = LoadPage(self.container, self.show_frame, self.load_mask_directory, self.load_ground_truth_directory, self.assign_class_names)
        self.load_page.grid(row=0, column=0, sticky="nsew")
        self.analysis_page = AnalysisPage(self.container, self.show_frame, self.show_next_image, self.show_previous_image)
        self.analysis_page.grid(row=0, column=0, sticky="nsew")
        self.export_page = ExportPage(self.container, self.show_frame, self.metrics_calculator.export.export)
        self.export_page.grid(row=0, column=0, sticky="nsew")
        self.metrics_page = MetricsPage(self.container, self.show_frame)
        self.metrics_page.grid(row=0, column=0, sticky="nsew")
        self.documentation_page = DocumentationPage(self.container, self.show_previous_frame)
        self.documentation_page.grid(row=0, column=0, sticky="nsew")
                
        (self.thumbnail_mask_canvas, self.thumbnail_mask_frame, self.scroll_y_mask), \
        (self.thumbnail_pred_canvas, self.thumbnail_pred_frame, self.scroll_y_pred) = self.init_thumbnails()

        # Add Treeview to display existing class aliases
        self.alias_frame = tk.Frame(root)
        self.class_alias_tree = ttk.Treeview(self.alias_frame, columns=("Color", "Alias"), show="headings")
        self.class_alias_tree.heading("Color", text="Class Color")
        self.class_alias_tree.heading("Alias", text="Class Alias")
        
        self.class_alias_tree.column("Color", width=100, anchor="center")  # Center-aligned
        self.class_alias_tree.column("Alias", width=100, anchor="w")   

        # Scrollbar for Treeview
        self.tree_scrollbar = ttk.Scrollbar(self.alias_frame, orient="vertical", command=self.class_alias_tree.yview)
        self.class_alias_tree.configure(yscrollcommand=self.tree_scrollbar.set)
        
        # Pack Treeview and Scrollbar inside the frame
        self.class_alias_tree.pack(side="left", fill="both", expand=True)
        self.tree_scrollbar.pack(side="right", fill="y")
        
        self.update_class_aliases(self.class_manager.color_to_name)

        self.frames = {}

        # Add LoadPage to frames
        self.frames["LoadPage"] = self.load_page
        self.frames["AnalysisPage"] = self.analysis_page
        self.frames["ExportPage"] = self.export_page
        self.frames["DocumentationPage"] = self.documentation_page
        self.frames["MetricsPage"] = self.metrics_page

        self.show_frame("LoadPage")
        
        self.root.update_idletasks()

        self.create_widgets(str(self.current_page))
            
        self.root.update_idletasks()
        print("Canvas dimensions:", self.canvas.winfo_width(), self.canvas.winfo_height())
        print("Legend dimensions:", self.legend_frame.winfo_width(), self.legend_frame.winfo_height())

        # Bind specific mouse wheel events to the respective widgets
        self.metrics_calculator.table.bind("<MouseWheel>", self.scroll_table)
        self.canvas.bind("<MouseWheel>", self.zoom_canvas)

        
    def init_thumbnails(self):
        """
        Initializes scrollable thumbnail frames for displaying mask images (ground truth) 
        and predicted images in the application.
        Returns:
            tuple: A tuple containing:
                - (canvas_mask, scroll_frame_mask, scroll_y_mask): Elements for the mask image scrollable frame.
                - (canvas_pred, scroll_frame_pred, scroll_y_pred): Elements for the predicted image scrollable frame.
        """
        
        print("Initializing Thumbnails...")  # Debugging print

        # Scrollable frame for mask images (ground truth)
        self.canvas_mask = tk.Canvas(self.root, bg="#cce3fd", width=400, height=200)  
        self.scroll_y_mask = tk.Scrollbar(self.root, orient="vertical", command=self.canvas_mask.yview)
        self.scroll_frame_mask = tk.Frame(self.canvas_mask) 

        self.scroll_frame_mask.bind(
            "<Configure>",
            lambda e: self.canvas_mask.configure(scrollregion=self.canvas_mask.bbox("all"))
        )

        self.canvas_mask.create_window((0, 0), window=self.scroll_frame_mask, anchor="nw")
        self.canvas_mask.configure(yscrollcommand=self.scroll_y_mask.set)

        print("Created mask image scrollable frame")  # Debugging print

        # Scrollable frame for predicted images
        self.canvas_pred = tk.Canvas(self.root, bg="#cce3fd", width=320, height=200)  
        self.scroll_y_pred = tk.Scrollbar(self.root, orient="vertical", command=self.canvas_pred.yview)
        self.scroll_frame_pred = tk.Frame(self.canvas_pred) 

        self.scroll_frame_pred.bind(
            "<Configure>",
            lambda e: self.canvas_pred.configure(scrollregion=self.canvas_pred.bbox("all"))
        )

        self.canvas_pred.create_window((0, 0), window=self.scroll_frame_pred, anchor="nw")
        self.canvas_pred.configure(yscrollcommand=self.scroll_y_pred.set)

        print("Created predicted image scrollable frame")  # Debugging print

        return (self.canvas_mask, self.scroll_frame_mask, self.scroll_y_mask), (self.canvas_pred, self.scroll_frame_pred, self.scroll_y_pred)

        
    def update_image_dropdown(self):
        """
        Updates the dropdown menu for image selection in the UI.
        This function populates the dropdown menu with the filenames of the 
        images stored in the `mask_images` attribute. If no images are available, 
        the dropdown is cleared. Additionally, it sets the currently selected 
        image in the dropdown to the index specified by `current_idx`.
        Attributes:
        - self.mask_images (list): A list of file paths to the mask images.
        - self.image_selector (widget): The dropdown widget for image selection.
        - self.current_idx (int): The index of the currently selected image.
        Behavior:
        - Clears the dropdown if `mask_images` is not defined or empty.
        - Populates the dropdown with the base filenames of the images.
        - Sets the current selection to the image at `current_idx` if available.
        """
        
        if not hasattr(self, 'mask_images') or not self.mask_images:
            self.image_selector['values'] = []  # Clear dropdown
            return

        # Use image filenames or indices as dropdown values
        image_names = [os.path.basename(image) for image in self.mask_images]
        self.image_selector['values'] = image_names
        if image_names:
            self.image_selector.current(self.current_idx)  # Select the first image by default

    def select_image_from_dropdown(self, event=None):
        """
        Handles the selection of an image from a dropdown menu.
        This method is triggered when an image is selected from the dropdown.
        It updates the current image index based on the selected image name,
        and refreshes the class selector, class visualization, and image metrics.
        If the selected image is not found, an error message is displayed.
        Args:
            event (optional): The event that triggered the method. Defaults to None.
        """
        
        selected_image_name = self.image_selector.get()
        if not selected_image_name:
            return

        # Find the index of the selected image
        try:
            self.current_idx = [
                os.path.basename(image) for image in self.mask_images
            ].index(selected_image_name)
            self.update_class_selector()
            self.update_class_visualization()
            self.update_current_image_metrics()
        except ValueError:
            messagebox.showerror("Error", f"Image '{selected_image_name}' not found in the loaded images.")
        
        
    def initialize_checkboxes(self, parent):
        """
        Creates and initializes a labeled frame containing checkboxes for selecting metrics to export.
        Args:
            parent (tk.Widget): The parent widget in which the checkbox frame will be placed.
        Returns:
            ttk.LabelFrame: The frame containing the checkboxes.
        """
        
        checkbox_frame = ttk.LabelFrame(parent, text="Select Metrics to Export", padding=10)
        checkbox_frame.place(relx=0.46, rely=0.14, relwidth=0.15, relheight=0.6)  # Adjust placement as needed

        for metric, var in self.checkbox_states.items():
            checkbox = ttk.Checkbutton(checkbox_frame, text=metric, variable=var)
            checkbox.pack(anchor="w", pady=5)
        
        return checkbox_frame
    
    def update_current_image_metrics(self):
        """
        Updates the metrics table for the currently selected image by calculating 
        various segmentation metrics for each unique class in the image.
        This function performs the following steps:
        1. Checks if mask and ground truth images are available. If not, it exits early.
        2. Loads the current mask and ground truth images based on the current index.
        3. Detects unique classes present in the current image using the analyzer.
        4. Clears the existing metrics table to prepare for new data.
        5. Iterates over each unique class (excluding the background class) and:
           - Computes segmentation metrics for the class.
           - Updates the metrics table with the computed values.
        Metrics calculated include:
        - Dice coefficient
        - Intersection over Union (IoU)
        - Accuracy
        - Precision
        - Recall
        - Specificity
        - Fallout
        - False Negative Rate (FNR)
        - Volume similarity
        - Area under the curve (AUC)
        - Global Consistency Error (GCE)
        - Cohen's Kappa score
        - Average Hausdorff Distance (AHD)
        - Average Symmetric Surface Distance (ASSD)
        - Dice Similarity Coefficient (DSC)
        - Boundary IoU score
        Note:
        - The background class (assumed to be `(0, 0, 0)`) is skipped during metric computation.
        - The metrics table is updated dynamically for each class.
        Returns:
            None
        """
        
        if not self.mask_images or not self.gt_images:
            return

        current_mask_path = self.mask_images[self.current_idx]
        current_gt_path = self.gt_images[self.current_idx]

        # Load the mask and ground truth images
        pred = cv2.imread(current_mask_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(current_gt_path, cv2.IMREAD_GRAYSCALE)

        # Get unique classes in the current image
        unique_classes =  self.analyzer.detect_classes(current_mask_path, current_gt_path, False, None, None)  #np.unique(np.concatenate((pred, gt)))

        # Clear the table before updating
        self.current_image_metrics_calculator.table.delete(*self.current_image_metrics_calculator.table.get_children())

        # Calculate and add metrics for each class
        for target_class in unique_classes:
            if target_class == (0, 0, 0):  # Skip background class
                continue

            metrics = self.current_image_metrics_calculator.compute_metrics(pred, gt, target_class)
            
            self.current_image_metrics_calculator.table.column("Image", width=0, minwidth=0, stretch=tk.NO)
            self.current_image_metrics_calculator.table.insert("", "end", values=(
                "",
                f"Class {self.class_manager.get_class_name(target_class)}",
                metrics["dice"],
                metrics["iou"],
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["specificity"],
                metrics["fallout"],
                metrics["fnr"],
                metrics["vol_similarity"],
                metrics["auc"],
                metrics["gce_score"],
                metrics["kappa_score"],
                metrics["ahd_score"],
                metrics["assd_score"],
                metrics["dsc_score"],
                metrics["boundary_iou_score"],
                "-"
            ))      

    def show_frame(self, page_name):
        """
        Display the specified frame (page) in the application.
        This method manages the navigation between different pages of the application.
        It ensures that certain conditions are met before allowing navigation to specific pages,
        such as requiring analysis to be completed before accessing results-related pages.
        Args:
            page_name (str): The name of the page to display.
        Behavior:
            - If the page is in the `frames` dictionary, it attempts to switch to it.
            - Prevents navigation to "AnalysisPage" if `mask_images` and `gt_images` are not available.
            - Prevents navigation to "MetricsPage" or "ExportPage" if analysis has not been completed.
            - Displays an error message if navigation conditions are not met.
            - Raises the frame to the top and creates widgets for the current page if navigation is successful.
            - Automatically starts analysis when navigating to "AnalysisPage".
        """
        
        if page_name in self.frames:
            if hasattr(self, 'current_page') and self.current_page:  # Remember the current page
                self.previous_page = self.current_page
            self.current_page = page_name
            frame = self.frames[page_name]
            
             
            if str(self.current_page) == "AnalysisPage" and not self.mask_images and not self.gt_images:
                self.current_page = self.previous_page
            
            elif (str(self.current_page) == "MetricsPage" or str(self.current_page) == "ExportPage") and self.analysis_done == False:
                messagebox.showerror("Error", "Analysis must be started before the results are available.")
                self.current_page = self.previous_page
            else:
                frame.tkraise()
                self.create_widgets(str(self.current_page))
                
            if page_name == "AnalysisPage":
                self.run_analysis()

    def show_previous_frame(self):
        """
        Navigate to and display the previously visited frame in the application.
        This method checks if there is a previously visited page (`previous_page`) 
        and, if it exists, calls the `show_frame` method to display it.
        """
        
        if self.previous_page:
            self.show_frame(self.previous_page)

    def scroll_table(self, event):
        """
        Handles the scrolling of a table widget in response to mouse wheel events.
        Args:
            event: The event object containing information about the mouse wheel action.
        """
        self.metrics_calculator.table.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def zoom_canvas(self, event):
        """
        Handles the zooming functionality for the canvas.
        This method is triggered by an event and delegates the zooming action
        to the `zoom_image` method of the `image_display` object.
        Args:
            event: The event object containing information about the zoom action.
        """
        
        self.image_display.zoom_image(event)

    def handle_mouse_wheel(self, event):
        """
        Handles mouse wheel events to enable scrolling or zooming based on the focused widget.
        If the metrics calculator table is focused, the mouse wheel event is passed to the table
        for vertical scrolling. Otherwise, the event is passed to the image display for zooming.
        Args:
            event: The mouse wheel event containing information about the scroll direction.
        """
        
        focused_widget = self.root.focus_get()
        if focused_widget == self.metrics_calculator.table:  # Table is focused
            # Pass the scroll event to the table
            self.metrics_calculator.table.yview_scroll(-1 if event.delta > 0 else 1, "units")
        else:  # Canvas or other widget is focused
            # Pass the event to the image display for zooming
            self.image_display.zoom_image(event)

    def create_widgets(self, page_name):
        """
        Create and configure widgets for different pages in the application.
        This method dynamically sets up the layout and widgets based on the 
        provided `page_name`. It handles the initialization and placement of 
        widgets for the following pages:
        - AnalysisPage: Configures dropdowns, frames, and buttons for analyzing 
          images and datasets.
        - MetricsPage: Displays metrics in a dedicated frame.
        - ExportPage: Provides options for exporting data, including format 
          selection, file path configuration, and preview generation.
        - LoadPage: Sets up canvases, scrollbars, and a treeview for loading 
          and displaying image thumbnails and aliases.
        Args:
            page_name (str): The name of the page for which widgets should be 
            created. Valid values are "AnalysisPage", "MetricsPage", 
            "ExportPage", and "LoadPage".
        """
        
        if page_name == "AnalysisPage":
            # Configure and display the legend frame
            self.legend_frame.place(relx=0.455, rely=0.525, width=150, height=110)
            self.legend_frame.tkraise()

            # Add class selector dropdown for selecting classes
            self.class_selector = ttk.Combobox(self.container, state="readonly")
            self.class_selector.place(relx=0.71, rely=0.549, width=225, height=20)
            self.class_selector.bind("<<ComboboxSelected>>", self.update_class_visualization)
            
            # Add the dropdown for selecting specific images
            self.image_selector = ttk.Combobox(self.container, state="readonly")
            self.image_selector.place(relx=0.71, rely=0.618, width=225, height=20)  # Adjust placement
            self.image_selector.bind("<<ComboboxSelected>>", self.select_image_from_dropdown)
            
            # Populate the image dropdown with available images
            self.update_image_dropdown()
            
            # Configure and display the metrics frames
            self.current_image_metrics_frame.place(relx=0.46, rely=0.115, width=755, height=365)
            self.current_image_metrics_frame.tkraise()
            self.dataset_metrics_frame.place(relx=0.016, rely=0.79, width=1400, height=210)
            self.dataset_metrics_frame.tkraise()
            
            # Add and display the compare button
            self.compare_button.place(relx=0.46, rely=0.7, width=200, height=30)
            
            # Show the main canvas
            self.canvas.grid()
        else:
            # Hide the canvas and compare button when not on the AnalysisPage
            self.canvas.grid_remove()
            self.compare_button.place_forget()
        
        if page_name == "MetricsPage":
            # Configure and display the metrics frame
            self.metrics_frame.place(relx=0.013, rely=0.095, width=1400, height=900)
            self.metrics_frame.tkraise()
        else:
            # Hide the metrics frame when not on the MetricsPage
            self.metrics_frame.place_forget()
            
        if page_name == "ExportPage":
            # Initialize and display the checkboxes for metric selection
            self.checkbox_frame = self.initialize_checkboxes(self.container)
            
            # Add label and input field for image selection
            ttk.Label(self.container, text="Images (e.g., 1;3;5-8):").place(relx=0.62, rely=0.75, width=135, height=20)
            self.image_selection_input = ttk.Entry(self.container)
            self.image_selection_input.place(relx=0.46, rely=0.75, width=220, height=25)
            
            # Add dropdown for selecting export format (JSON or CSV)
            self.format_selector = ttk.Combobox(self.container, textvariable=self.metrics_calculator.export.export_format, state="readonly", values=["json", "csv"])
            self.format_selector.bind("<<ComboboxSelected>>", lambda e: self.update_preview())
            self.format_selector.place(relx=0.7, rely=0.143, width=200, height=20)  # Adjust placement
            
            # Add a button to select the export file path
            self.path_button = ttk.Button(
            self.container,
            text="No file selected",
            style="Path.TButton"
            )
            self.path_button.configure(command=lambda: self.metrics_calculator.export.set_destination_path(self.path_button))
            self.path_button.place(relx=0.7, rely=0.23, width=320, height=25)

            # Configure the style for the path button
            style = ttk.Style()
            style.configure(
            "Path.TButton",
            relief="flat",  # Flat border to mimic a label
            anchor="w",  # Align text to the left
            background="white"
            )
            
            # Add a text widget for previewing the export data
            self.preview_text = tk.Text(self.container, wrap="word", state="normal", height=20, width=60)
            self.preview_text.place(relx=0.014, rely=0.14, width=620, height=843)
            self.preview_text.config(state="disabled")  # Make it read-only initially
            
            # Add a button to regenerate the preview
            regenerate_button = ttk.Button(self.container, text="Regenerate Preview", command=self.update_preview)
            regenerate_button.place(relx=0.465, rely=0.8, width=200, height=30)

            # Generate the initial preview
            self.update_preview()
            
        if page_name == "LoadPage":
            # Configure and display the thumbnail canvases and scrollbars
            self.thumbnail_pred_canvas.place(relx=0.407, rely=0.255, width=400, height=505)
            self.thumbnail_mask_canvas.place(relx=0.073, rely=0.255, width=400, height=505)
            self.scroll_y_mask.place(relx=0.35, rely=0.255, width=20, height=505)
            self.scroll_y_pred.place(relx=0.68, rely=0.255, width=20, height=505)
            
            # Configure and display the alias frame for class aliases
            self.alias_frame.place(relx=0.73, rely=0.255, width=320, height=505)
        else:
            # Hide the thumbnail canvases, scrollbars, and alias frame when not on the LoadPage
            self.thumbnail_pred_canvas.place_forget()
            self.thumbnail_mask_canvas.place_forget()
            self.scroll_y_mask.place_forget()
            self.scroll_y_pred.place_forget()
            self.alias_frame.place_forget()

    def show_loading_window(self):
        """
        Displays a loading window with a progress bar and a message indicating that metrics are being calculated.
        This method creates a new top-level window with a title, a message, and an indeterminate progress bar.
        It is intended to inform the user that a background analysis is in progress. The window allows the user
        to continue interacting with the main interface while the analysis completes.
        Attributes:
            self.loading_window (tk.Toplevel): The top-level window displaying the loading message and progress bar.
            self.analysis_done (bool): A flag indicating whether the analysis has been completed. If False, the
                                       loading window is displayed and the flag is set to True.
        """
        
        if self.analysis_done == False:
            self.loading_window = tk.Toplevel(self.root)
            self.loading_window.title("Calculating...")
            self.loading_window.geometry("500x150")
            self.loading_window.resizable(False, False)

            label = tk.Label(self.loading_window, text="Calculating metrics...\nPlease wait. \nYou can work with interface, data will be loaded in the meantime", font=("Arial", 12))
            label.pack(pady=20)

            # Add a progress bar
            progress_bar = ttk.Progressbar(self.loading_window, mode='indeterminate')
            progress_bar.pack(pady=10, padx=20, fill="x")
            self.analysis_done = True
            progress_bar.start()

    def open_comparison_window(self):
        """
        Opens a new window to compare the current mask image with the ground truth image.
        This function checks if mask and ground truth images are loaded. If not, it displays an error message.
        Otherwise, it creates a new popup window where the current mask image and ground truth image are displayed
        side by side for comparison.
        Raises:
            Displays a messagebox error if no images are loaded.
        Notes:
            - The images are resized to 350x350 pixels using LANCZOS resampling for display purposes.
            - The popup window is set to a size of 800x400 pixels.
            - The images are displayed with labels indicating "Mask" and "Ground Truth".
        """

        if not self.mask_images or not self.gt_images:
            messagebox.showerror("Error", "No images loaded.")
            return

        # Create a new Toplevel window
        popup = tk.Toplevel(self.root)
        popup.title("Image Comparison")
        popup.geometry("800x400")  # Set an appropriate size

        # Load images
        mask_img_path = self.mask_images[self.current_idx]
        gt_img_path = self.gt_images[self.current_idx]

        mask_img = Image.open(mask_img_path).resize((350, 350), Image.Resampling.LANCZOS)
        gt_img = Image.open(gt_img_path).resize((350, 350), Image.Resampling.LANCZOS)

        # Convert to Tkinter format
        mask_img_tk = ImageTk.PhotoImage(mask_img)
        gt_img_tk = ImageTk.PhotoImage(gt_img)

        # Create labels to display images
        mask_label = tk.Label(popup, image=mask_img_tk, text="Mask", compound="top")
        mask_label.image = mask_img_tk  # Keep reference
        mask_label.pack(side="left", padx=10, pady=10)

        gt_label = tk.Label(popup, image=gt_img_tk, text="Ground Truth", compound="top")
        gt_label.image = gt_img_tk  # Keep reference
        gt_label.pack(side="right", padx=10, pady=10)

            
    def update_class_aliases(self, color_to_name):
        """
        Updates the class alias Treeview with the provided color-to-name mapping.
        This method clears the existing items in the Treeview widget and repopulates
        it with the current mapping of colors to their respective aliases.
        Args:
            color_to_name (dict): A dictionary where keys are colors (e.g., strings or tuples)
                                  and values are their corresponding alias names (str).
        """
        
        # Clear existing Treeview items
        for item in self.class_alias_tree.get_children():
            self.class_alias_tree.delete(item)

        # Populate Treeview with current class aliases
        for color, alias in color_to_name.items():
            self.class_alias_tree.insert("", "end", values=(color, alias))

    def assign_class_names(self):
        """
        Assigns class names to detected classes in the mask and ground truth images.
        This function checks if mask and ground truth images are loaded. If not, it displays an error message.
        It then detects classes and their associated colors using the analyzer and opens a dialog
        to assign class names using the class manager.
        Raises:
            tkinter.messagebox.showerror: If no images are loaded before assigning class names.
        """
        
        if not self.mask_images or not self.gt_images:
            messagebox.showerror("Error", "Please load images before assigning class names.")
            return
        classes_colors = self.analyzer.detect_classes(None, None, True, self.mask_images, self.gt_images)
        self.class_manager.open_class_name_dialog(classes_colors)
        
    def parse_image_selection(self):
        """
        Parse the input from the image selection field to get the list of selected image indices.
        Example input: "1;3;5-8"
        Returns:
            List of selected indices (zero-based).
        """
        input_text = self.image_selection_input.get().strip()
        if not input_text:
            return []  # No selection
        
        selected_indices = set()
        try:
            # Split input by semicolon
            parts = input_text.split(";")
            for part in parts:
                if "-" in part:  # Range case
                    start, end = map(int, part.split("-"))
                    selected_indices.update(range(start - 1, end))  # Convert to zero-based index
                else:  # Single index case
                    selected_indices.add(int(part) - 1)  # Convert to zero-based index
        except ValueError:
            messagebox.showerror("Error", "Invalid syntax. Use format like '1;3;5-8'.")
            return []

        # Validate indices are within range
        valid_indices = [i for i in selected_indices if 0 <= i < len(self.mask_images)]
        if len(valid_indices) != len(selected_indices):
            messagebox.showwarning("Warning", "Some indices are out of range and will be ignored.")
        
        return valid_indices
    
    def display_thumbnails(self, filepaths, scroll_frame):
        """
        Display image thumbnails and their filenames in a scrollable frame.
        This function takes a list of file paths to images and displays their thumbnails
        along with their filenames in a specified scrollable frame. The thumbnails are
        arranged in a grid with a maximum of two columns.
        Args:
            filepaths (list): A list of file paths to the images to be displayed.
            scroll_frame (tk.Frame): The scrollable frame where thumbnails and filenames
                                     will be displayed.
        Returns:
            list: A list of references to the PhotoImage objects to prevent garbage collection.
        """
        
        self.scroll_frame = scroll_frame
        for widget in self.scroll_frame.winfo_children():  # Remove old thumbnails
            widget.destroy()

        max_columns = 2  # Limit to 2 columns
        row, col = 0, 0

        self.image_refs = []  # Store references to prevent garbage collection

        for filepath in filepaths:
            img = Image.open(filepath)
            img = img.resize((150, 150), Image.Resampling.LANCZOS)  # Thumbnail size
            img_tk = ImageTk.PhotoImage(img)

            # Keep a reference
            self.image_refs.append(img_tk)

            # Create image label
            img_label = tk.Label(self.scroll_frame, image=img_tk)
            img_label.grid(row=row, column=col, padx=23, pady=10)

            # Create text label (filename)
            text_label = tk.Label(self.scroll_frame, text=os.path.basename(filepath), font=("Arial", 10))
            text_label.grid(row=row+1, column=col, pady=5)

            # Move to the next column
            col += 1
            if col >= max_columns:  # Move to new row if max columns reached
                col = 0
                row += 2  # Move 2 rows down (one for image, one for text)
                
        self.scroll_frame.update_idletasks()
        return self.image_refs

    
    def update_preview(self):
        """
        Updates the preview text widget with the exported data in the selected format.
        This function retrieves the selected export format (JSON or CSV) from the 
        metrics calculator, generates a preview of the exported data, and displays 
        it in the preview text widget. The widget is temporarily made editable to 
        update its content and then set back to read-only mode.
        Raises:
            ValueError: If an invalid export format is selected.
        """

        self.preview_text.config(state="normal")  # Enable editing to update content
        self.preview_text.delete("1.0", tk.END)  # Clear existing content

        export_format = self.metrics_calculator.export.get_selected_format()
        if export_format == "json":
            preview_data = self.metrics_calculator.export.exportJson(preview=True)
        elif export_format == "csv":
            preview_data = self.metrics_calculator.export.exportCsv(preview=True)
        else:
            preview_data = "Invalid format selected."

        self.preview_text.insert(tk.END, preview_data)
        self.preview_text.config(state="disabled")  # Make read-only again
        
    def create_canvas(self):
        """
        Creates a canvas widget with specified dimensions and background color.
        The canvas is initially hidden and positioned in the top-left corner of the grid layout.
        Returns:
            tkinter.Canvas: The created canvas widget.
        """
        
        # Create the canvas with specific dimensions and background
        canvas = tk.Canvas(self.root, width=615, height=615, bg="white")
        canvas.grid(row=0, column=0, padx=21, pady=120, sticky="nw")  # Place in the top-left corner
        canvas.grid_remove()  # Hide the canvas initially
        return canvas


    def load_mask_directory(self):
        """
        Load mask images from a selected directory and display their thumbnails.
        This function opens a file dialog to allow the user to select a directory
        containing mask images. The images are then loaded, their thumbnails are
        displayed in the specified frame, and the analysis status is reset.
        Steps:
        1. Opens a directory selection dialog.
        2. Loads mask images from the selected directory.
        3. Displays thumbnails of the loaded images in the thumbnail frame.
        4. Resets the analysis status in the metrics calculator.
        Attributes:
            self.mask_images (list): List of loaded mask images.
            self.pred_refs (list): List of references to the displayed thumbnails.
            self.metrics_calculator.analysis_done (bool): Reset to False to indicate
                that analysis needs to be performed again.
        """
        
        directory = filedialog.askdirectory(title="Select Mask Directory")
        self.mask_images = DirectoryLoader.load_images_from_directory(directory)
        
        self.pred_refs = self.display_thumbnails(self.mask_images, self.thumbnail_pred_frame)
        
        self.metrics_calculator.analysis_done = False

    def load_ground_truth_directory(self):
        """
        Opens a dialog for the user to select a directory containing ground truth images,
        loads the images from the selected directory, displays their thumbnails, and resets
        the analysis status in the metrics calculator.
        Steps:
        1. Prompts the user to select a directory using a file dialog.
        2. Loads images from the selected directory using the DirectoryLoader utility.
        3. Displays thumbnails of the loaded images in the specified thumbnail frame.
        4. Resets the `analysis_done` flag in the metrics calculator to indicate that
           analysis needs to be performed again.
        Attributes:
        - self.gt_images: List of images loaded from the selected directory.
        - self.mask_refs: References to the displayed thumbnails for the loaded images.
        - self.metrics_calculator.analysis_done: Boolean flag indicating the analysis status.
        """
        
        directory = filedialog.askdirectory(title="Select Ground Truth Directory")
        self.gt_images = DirectoryLoader.load_images_from_directory(directory)
        
        self.mask_refs = self.display_thumbnails(self.gt_images, self.thumbnail_mask_frame)
        self.metrics_calculator.analysis_done = False

    def run_analysis(self):
        """
        Run the analysis process for comparing mask images and ground truth images.
        This function validates that both mask and ground truth image directories are loaded 
        and contain matching image counts. If validation passes, it displays a loading window 
        and initiates the analysis process in a separate thread to avoid blocking the main UI thread.
        Note:
            - The analysis process is executed in a background thread for better performance.
            - A periodic call to `_run_analysis` is scheduled using `self.root.after` to ensure 
              the loading window remains responsive.
        Raises:
            tkinter.messagebox.showerror: If the mask and ground truth directories are not loaded 
            or their image counts do not match.
        """
        
        if not self.mask_images or not self.gt_images or len(self.mask_images) != len(self.gt_images):
            messagebox.showerror("Error", "Please load both mask and ground truth directories with matching image counts.")
            return
        self.show_loading_window()

        # Run the analysis in a separate thread
        analysis_thread = threading.Thread(target=self._run_analysis, daemon=True)
        analysis_thread.start()  # Start background thread


        self.root.after(100, self._run_analysis)
        #added because of the popup window
    def _run_analysis(self):
        """
        Perform analysis by updating image metrics, class selector, and visualization, 
        and calculating and displaying metrics for mask and ground truth images.
        This method updates the current image metrics, refreshes the class selector 
        and visualization, and invokes the metrics calculator to compute and display 
        metrics based on the provided mask and ground truth images.
        """

        self.update_current_image_metrics()
        self.update_class_selector()
        self.update_class_visualization()
        self.metrics_calculator.calculate_and_display_metrics(self.mask_images, self.gt_images)
 # Ensure safe UI update
            
    def close_loading_window(self):
        """
        Closes the loading window if it exists.
        This method checks if the 'loading_window' attribute exists and is a valid 
        window. If so, it destroys the window to close it.
        """
    
        if hasattr(self, 'loading_window') and self.loading_window.winfo_exists():
            self.loading_window.destroy()


    def update_class_selector(self):
        """
        Updates the class selector combobox with the detected unique classes 
        from the current mask and ground truth images.
        This function retrieves the unique classes detected by the analyzer 
        from the current mask and ground truth images. It then formats the 
        class names with their respective grayscale ID (gID) and updates the 
        combobox values. If classes are detected, the first class is selected 
        by default and set as the current class in the analyzer. If no classes 
        are detected, the combobox is cleared.
        """
        
        unique_classes = self.analyzer.detect_classes(
            self.mask_images[self.current_idx],
            self.gt_images[self.current_idx], False, None, None
        )

        class_names = [
            f"Class {self.class_manager.get_class_name(c)}: gID: {str(int(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]))}" for c in unique_classes
        ]

        # Update the combobox values
        self.class_selector['values'] = class_names
        if class_names:
            self.class_selector.current(0)
            self.analyzer.current_class = class_names[0]
        else:
            self.class_selector.set("")


    def update_class_visualization(self, event=None):
        """
        Updates the visualization of a selected class by highlighting its 
        misclassification in the current mask and ground truth images.
        This function retrieves the selected class from the dropdown menu, parses 
        its RGB color tuple, and uses the analyzer to generate a visualization of 
        misclassified regions for the selected class. The visualization is then 
        displayed in the application's image display widget.
        Args:
            event (optional): An event object, typically passed when the function 
                              is triggered by a GUI event. Defaults to None.
        Returns:
            None: The function updates the GUI and does not return any value.
        Raises:
            Displays an error message if the selected class format is invalid.
        """
        
        if not self.mask_images or not self.gt_images:
            return

        selected_class = self.class_selector.get()
        if selected_class:
            # Extract the class color tuple from the dropdown
            try:
                color_str = selected_class.split(":")[1].strip()
                target_class = tuple(map(int, color_str.strip("()").split(", ")))
            except (IndexError, ValueError) as e:
                messagebox.showerror("Error", f"Invalid class format: {selected_class}")
                return

            # Visualize the selected class
            visualization = self.analyzer.visualize_misclassification(
                self.mask_images[self.current_idx],
                self.gt_images[self.current_idx],
                target_class=target_class
            )
            print(f"Selected class: {selected_class}")
            print(f"Target class RGB: {target_class}")

            if visualization is None:
                print("Error: Visualization is None.")  # Debugging
            else:
                print(f"Visualization created: shape={visualization.shape}")  # Debugging
            self.image_display.set_image(visualization)
        
    def show_next_image(self):
        """
        Advances to the next image in the list of mask images, updates the current index, 
        and refreshes all relevant UI components and metrics.
        This method performs the following actions:
        - Increments the current image index if it is not the last image.
        - Updates the class selector dropdown to reflect the new image.
        - Refreshes the class visualization for the new image.
        - Updates the metrics displayed for the current image.
        - Updates the image dropdown to reflect the new selection.
        """
        
        if self.current_idx < len(self.mask_images) - 1:
            self.current_idx += 1
            self.update_class_selector()
            self.update_class_visualization()
            self.update_current_image_metrics() 
            self.update_image_dropdown()

    def show_previous_image(self):
        """
        Navigate to the previous image in the sequence and update the UI components accordingly.
        This method decreases the current image index by one (if not already at the first image)
        and updates various UI elements such as the class selector, class visualization, 
        image metrics, and image dropdown to reflect the newly selected image.
        """
        
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_class_selector()
            self.update_class_visualization()
            self.update_current_image_metrics() 
            self.update_image_dropdown()

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()
