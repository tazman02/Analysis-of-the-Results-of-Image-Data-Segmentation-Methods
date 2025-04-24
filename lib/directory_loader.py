import os
from tkinter import messagebox

class DirectoryLoader:
    """
    A class for loading image files from a specified directory.

    Methods:
        load_images_from_directory(directory, extensions=('png', 'jpeg', 'jpg')):
            Loads and returns a sorted list of image file paths from the specified directory
            that match the given file extensions. If no valid image files are found, an error
            message is displayed.

            Args:
                directory (str): The path to the directory containing image files.
                extensions (tuple): A tuple of valid image file extensions to filter by. 
                                    Defaults to ('png', 'jpeg', 'jpg').

            Returns:
                list: A sorted list of file paths to the valid image files in the directory.
                      Returns an empty list if the directory is None or no valid files are found.
    """
    def load_images_from_directory(directory, extensions=('png', 'jpeg', 'jpg')):
        if directory:
            images = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extensions)])
            if not images:
                messagebox.showerror("Error", "No valid image files found in the selected directory.")
            return images
        return []