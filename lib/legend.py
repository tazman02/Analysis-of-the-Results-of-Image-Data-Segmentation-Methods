import tkinter as tk

class Legend:
    """
    A class to create a legend with checkboxes for toggling the visibility of different categories.
    Attributes:
        update_callback (function): A callback function that is triggered when a checkbox state changes.
        show_tp (tk.BooleanVar): A tkinter variable to track the state of the "True Positive" checkbox.
        show_tn (tk.BooleanVar): A tkinter variable to track the state of the "True Negative" checkbox.
        show_fp (tk.BooleanVar): A tkinter variable to track the state of the "False Positive" checkbox.
        show_fn (tk.BooleanVar): A tkinter variable to track the state of the "False Negative" checkbox.
        legend_items (list): A list of tuples containing the label, color, and tkinter variable for each legend item.
    Methods:
        __init__(root, update_callback):
            Initializes the Legend instance and creates the legend items.
        create_legend(root):
            Creates and displays the checkboxes for the legend items in the given root widget.
    """
    def __init__(self, root, update_callback):
        self.update_callback = update_callback
        self.show_tp = tk.BooleanVar(value=True)
        self.show_tn = tk.BooleanVar(value=True)
        self.show_fp = tk.BooleanVar(value=True)
        self.show_fn = tk.BooleanVar(value=True)

        self.legend_items = [
            ("TP (True Positive)", "green", self.show_tp),
            ("TN (True Negative)", "blue", self.show_tn),
            ("FP (False Positive)", "red", self.show_fp),
            ("FN (False Negative)", "yellow", self.show_fn)
        ]
        self.create_legend(root)

    def create_legend(self, root):
        for label, color, var in self.legend_items:
            checkbox = tk.Checkbutton(root, text=label, variable=var, command=self.update_callback,
                                    selectcolor=color, indicatoron=True, background="white") 
            checkbox.pack(anchor="w", pady=2)