import os
import tkinter as tk
import tkinter.ttk as ttk
import json

class ClassManager:
    """
    ClassManager is responsible for managing the mapping between colors and class names
    used in the application. It provides functionality to load, save, and modify these
    mappings, as well as interact with the user through a GUI dialog for assigning class
    names to colors.
    Attributes:
        color_to_name (dict): A dictionary mapping color tuples (R, G, B) to class names.
        app_instance: The main application instance that uses this class.
        file_path (str): The path to the JSON file where class names are stored.
        metrics_calculator: An optional reference to a metrics calculator for triggering
            re-analysis after class names are updated.
    Methods:
        __init__(app_instance, file_path="program/class_names.json"):
            Initializes the ClassManager with the application instance and file path.
        set_metrics_calculator(metrics_calculator):
            Sets the metrics calculator instance for triggering re-analysis.
        get_class_name(color):
            Retrieves the class name for a given color tuple. If no name is found, returns
            "Unnamed" with the color tuple.
        add_class_name(color, name):
            Adds or updates the class name for a given color tuple and saves the mapping
            to the JSON file.
        save_class_names():
            Saves the current color-to-name mapping to a JSON file.
        load_class_names():
            Loads the color-to-name mapping from a JSON file if it exists. Handles JSON
            decoding errors gracefully.
        open_class_name_dialog(classes_colors):
            Opens a GUI dialog to allow the user to assign class names to a list of colors.
            Updates the color-to-name mapping and triggers re-analysis if necessary.
    """
    def __init__(self, app_instance, file_path="program/class_names.json"):
        self.color_to_name = {}
        self.app_instance = app_instance
        self.file_path = file_path
        self.load_class_names()

    def set_metrics_calculator(self, metrics_calculator):
        self.metrics_calculator = metrics_calculator 

    def get_class_name(self, color):
        color_tuple = tuple(map(int, color))
        return f"{self.color_to_name.get(color_tuple, 'Unnamed')}: {color_tuple}"

    def add_class_name(self, color, name):
        color_tuple = tuple(map(int, color))
        self.color_to_name[color_tuple] = name
        self.save_class_names()

    def save_class_names(self):
        """ Save the color-to-name mapping to a JSON file. """
        # Convert tuple keys to strings for JSON compatibility
        data_to_save = {str(k): v for k, v in self.color_to_name.items()}
        with open(self.file_path, "w") as file:
            json.dump(data_to_save, file)

    def load_class_names(self):
        """ Load class names from a JSON file if it exists. """
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                try:
                    data = json.load(file)
                    # Convert string keys back to tuples
                    self.color_to_name = {
                        tuple(map(int, k.strip("()").split(", "))): v for k, v in data.items()
                    }
                except json.JSONDecodeError:
                    self.color_to_name = {}

    def open_class_name_dialog(self, classes_colors):
        """ Open a GUI dialog to assign class names to colors. """
        dialog = tk.Toplevel()
        dialog.title("Assign Class Names")

        frame = ttk.Frame(dialog)
        frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        entries = {}
        for i, color in enumerate(classes_colors):
            color_tuple = tuple(map(int, color))
            tk.Label(scrollable_frame, text=f"Class Color {color_tuple}").grid(row=i, column=0, sticky="w", padx=5, pady=5)

            entry = tk.Entry(scrollable_frame)
            entry.grid(row=i, column=1, padx=5, pady=5)
            entries[color_tuple] = entry

            # Pre-fill the entry if the color name was previously stored
            if color_tuple in self.color_to_name:
                entry.insert(0, self.color_to_name[color_tuple])

        def save_names():
            """ Save the entered class names and trigger re-analysis. """
            for color, entry in entries.items():
                name = entry.get()
                if name:
                    self.add_class_name(color, name)
            dialog.destroy()
            if self.metrics_calculator.analysis_done == True:
                self.metrics_calculator.analysis_done = False
                self.app_instance.run_analysis()
            self.app_instance.update_class_aliases(self.color_to_name)
            

        ttk.Button(dialog, text="Save", command=save_names).pack(pady=10)

        dialog.geometry("400x350")
        dialog.resizable(True, True)