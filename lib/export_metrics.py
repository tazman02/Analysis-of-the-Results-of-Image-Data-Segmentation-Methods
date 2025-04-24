import tkinter as tk
import csv
import json
from tkinter import filedialog, messagebox
from tkinter.filedialog import asksaveasfilename

class ExportMetrics:
    """
    ExportMetrics is a class designed to handle the export of metrics data from a table widget to either JSON or CSV formats. 
    It provides functionality for selecting specific metrics and images, previewing the export, and saving the data to a file.
    Attributes:
        table (tkinter.Treeview): The table widget containing the metrics data.
        app (object): The application instance, which provides access to image selection and checkbox states.
        export_format (tk.StringVar): A tkinter StringVar to store the selected export format (default is "json").
        file_path (str): The file path where the exported data will be saved.
    Methods:
        __init__(table, app):
            Initializes the ExportMetrics instance with the table widget and application instance.
        get_selected_images():
            Retrieves the indices of the selected images from the application.
        get_selected_metrics():
            Retrieves the list of selected metrics based on the application's checkbox states.
        export():
            Exports the data to the selected file format (JSON or CSV). Displays an error if no file path is selected.
        get_export_directory(filetype):
            Opens a file dialog to select the export file path based on the specified file type (JSON or CSV).
        exportCsv(separator=';', preview=False):
            Exports the selected metrics and images to a CSV file. Allows previewing the CSV content as a string.
        exportJson(preview=False):
            Exports the selected metrics and images to a JSON file. Allows previewing the JSON content as a string.
        get_selected_format():
            Returns the currently selected export format (JSON or CSV).
        set_destination_path(label):
            Opens a file dialog to set the destination path for the export file and updates the provided label widget.
    """
    def __init__(self, table, app):
        self.table = table
        self.app = app 
        self.export_format = tk.StringVar(value="json")  # Default format is JSON
        
    def get_selected_images(self):
        selected_indices = self.app.parse_image_selection()
        if not selected_indices:
            return []

        return selected_indices
    
    def get_selected_metrics(self):
        return [metric for metric, var in self.app.checkbox_states.items() if var.get()]
    
    def export(self):
        if not self.file_path:
            messagebox.showerror("Error", "No file path selected. Please select a destination path before exporting.")
            return
        
        if self.export_format.get() == "json":
            self.exportJson()
        elif self.export_format.get() == "csv":
            self.exportCsv()

    def get_export_directory(self, filetype):
        file_extension = ".json" if filetype == "json" else ".csv"
        filetypes = [("JSON files", "*.json")] if filetype == "json" else [("CSV files", "*.csv")]
        return asksaveasfilename(initialfile=f"metrics{file_extension}", defaultextension=file_extension, filetypes=filetypes, confirmoverwrite=True)

    def exportCsv(self, separator=';', preview=False):
        rows = [["Name", "Class", "Dice", "IoU", "Accuracy", "Precision", "Recall", "Specificity", "Fallout", "FNR", "Volumetric similarity", "AUC", "GCE score", "Kappa", "AHD", "ASSD", "DSC", "Boundary IoU", "AP"]]
        selected_metrics = self.get_selected_metrics()
        selected_metrics.append("Name")  # Ensure "Name" is always included
        
        # Include only selected columns
        header = [col for col in rows[0] if col in selected_metrics]
        rows[0] = header

        distinct_names = []
        all_rows = []

        # Collect all row data
        for row_id in self.table.get_children():
            row = self.table.item(row_id)['values']
            row_dict = {key: value for key, value in zip(
                ["Name", "Class", "Dice", "IoU", "Accuracy", "Precision", "Recall", "Specificity", "Fallout", "FNR", "Volumetric similarity", "AUC", "GCE score", "Kappa", "AHD", "ASSD", "DSC", "Boundary IoU", "AP"],
                row
            ) if key in selected_metrics}
            all_rows.append(row_dict)

            if row_dict["Name"] not in distinct_names:
                distinct_names.append(row_dict["Name"])

        # Filter based on selected IDs
        selected_distinct_names = []
        selected_ids = self.get_selected_images()
        if not selected_ids:
            selected_distinct_names.extend(distinct_names)
        else:
            for idx in selected_ids:
                if idx < len(distinct_names):  # Ensure index is within range
                    selected_distinct_names.append(distinct_names[idx])

        # Filter rows for export based on selected distinct names
        filtered_rows = [
            {key: row[key] for key in header}  # Include only selected columns
            for row in all_rows if row["Name"] in selected_distinct_names
        ]

        # Add filtered rows to CSV
        for row in filtered_rows:
            # Ensure values are strings and format metrics to six decimal places if numerical
            formatted_row = [
                f"{value:.6f}" if isinstance(value, (float, int)) and key != "Name" else str(value)
                for key, value in row.items()
            ]
            rows.append(formatted_row)

        if preview:
            # Return a string representation of the CSV
            return "\n".join([separator.join(map(str, row)) for row in rows])
        else:
            # Save to a file
            if not self.file_path:
                messagebox.showerror("Error", "No file path selected.")
                return
            with open(self.file_path, "w", newline='') as myfile:
                csvwriter = csv.writer(myfile, delimiter=separator)
                csvwriter.writerows(rows)
            messagebox.showinfo("Export Success", f"Data successfully exported to {self.file_path}.")
            
            if preview:
                # Return a string representation of the CSV
                return "\n".join([separator.join(map(str, row)) for row in rows])
            else:
                # Save to a file
                if not self.file_path:
                    messagebox.showerror("Error", "No file path selected.")
                    return
                with open(self.file_path, "w", newline='') as myfile:
                    csvwriter = csv.writer(myfile, delimiter=separator)
                    csvwriter.writerows(rows)
                messagebox.showinfo("Export Success", f"Data successfully exported to {self.file_path}.")

    def exportJson(self, preview=False):
        data = []
        selected_metrics = self.get_selected_metrics()
        selected_metrics.append("Name")
        selected_ids = self.get_selected_images()
        
        distinct_names = []
        for row_id in self.table.get_children():
            row = self.table.item(row_id)['values']
            row_dict = {key: value for key, value in zip(
                ["Name", "Class", "Dice", "IoU", "Accuracy", "Precision", "Recall", "Specificity", "Fallout", "FNR", "Volumetric similarity", "AUC", "GCE score", "Kappa", "AHD", "ASSD", "DSC", "Boundary IoU", "AP"],
                row
            ) if key in selected_metrics}
            
            if row_dict["Name"] not in distinct_names:
                distinct_names.append(row_dict["Name"])
            

        selected_distinct_names = []
        if not selected_ids:
            # If no specific selection, append all distinct names
            selected_distinct_names.extend(distinct_names)
        else:
            # Append only distinct names at positions specified by selected_ids
            for idx in selected_ids:
                if idx < len(distinct_names):  # Ensure index is within range
                    selected_distinct_names.append(distinct_names[idx])
                    
        for row_id in self.table.get_children():
            row = self.table.item(row_id)['values']
            row_dict = {key: value for key, value in zip(
                ["Name", "Class", "Dice", "IoU", "Accuracy", "Precision", "Recall", "Specificity", "Fallout", "FNR", "Volumetric similarity", "AUC", "GCE score", "Kappa", "AHD", "ASSD", "DSC", "Boundary IoU", "AP"],
                row
            ) if key in selected_metrics}
            
            # Append row_dict to data if the "Name" is in selected_distinct_names
            if row_dict["Name"] in selected_distinct_names:
                data.append(row_dict)

        if preview:
            # Return a string representation of the JSON
            return json.dumps(data, indent=4)
        else:
            # Save to a file
            if not self.file_path:
                messagebox.showerror("Error", "No file path selected.")
                return
            with open(self.file_path, "w") as json_file:
                json.dump(data, json_file, indent=4)
            messagebox.showinfo("Export Success", f"Data successfully exported to {self.file_path}.")

    def get_selected_format(self):
        return self.export_format.get()
    
    def set_destination_path(self, label):
        filetypes = [("JSON files", "*.json"), ("CSV files", "*.csv")]
        file_extension = self.export_format.get()
        initial_extension = ".json" if file_extension == "json" else ".csv"

        selected_path = filedialog.asksaveasfilename(
            title="Select Export File",
            defaultextension=initial_extension,
            filetypes=filetypes
        )

        if selected_path:
            self.file_path = selected_path
            label.config(text=selected_path)