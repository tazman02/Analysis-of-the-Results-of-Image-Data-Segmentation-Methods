import cv2
import numpy as np

class ImageAnalyzer:
    """
    ImageAnalyzer is a class designed for analyzing and visualizing image data, particularly for tasks involving 
    image segmentation and classification. It provides methods for detecting unique classes in images and 
    visualizing misclassification between predicted and ground truth images.
    Attributes:
        legend (object): An object containing visualization settings, such as toggles for showing true positives, 
            true negatives, false positives, and false negatives.
        class_manager (object): An object responsible for managing class-related operations.
        current_class (str): The currently selected class for analysis. Defaults to "All Classes".
    Methods:
        detect_classes(mask_image, gt_image, all_images=False, mask_images=None, gt_images=None):
            Detects unique classes in the provided mask and ground truth images. If `all_images` is True, it processes 
            multiple images specified by `mask_images` and `gt_images`. Returns a sorted list of unique RGB tuples 
            representing the detected classes.
            Args:
                mask_image (str): Path to the mask image file.
                gt_image (str): Path to the ground truth image file.
                all_images (bool, optional): Whether to process multiple images. Defaults to False.
                mask_images (list of str, optional): List of paths to mask image files. Required if `all_images` is True.
                gt_images (list of str, optional): List of paths to ground truth image files. Required if `all_images` is True.
            Returns:
                list of tuple: A sorted list of unique RGB tuples representing the detected classes.
            Raises:
                ValueError: If required images are not provided or if the images are invalid.
        visualize_misclassification(pred_path, gt_path, target_class=None):
            Visualizes misclassification between a predicted image and a ground truth image. The visualization highlights 
            true positives, true negatives, false positives, and false negatives using distinct colors. Optionally, a 
            specific target class can be analyzed.
            Args:
                pred_path (str): Path to the predicted image file.
                gt_path (str): Path to the ground truth image file.
                target_class (tuple, optional): An RGB tuple representing the target class to analyze. Defaults to None.
            Returns:
                numpy.ndarray: An RGB image visualizing the misclassification.
            Notes:
                - The visualization colors are defined as:
                    - True Positive (TP): Green
                    - True Negative (TN): Blue
                    - False Positive (FP): Red
                    - False Negative (FN): Yellow
                    - None: Black
                - The method uses the legend's visibility toggles to determine which categories to display.
    """
    def __init__(self, legend, class_manager):
        self.legend = legend
        self.class_manager = class_manager
        self.current_class = "All Classes"

    def detect_classes(self, mask_image, gt_image, all_images=False, mask_images=None, gt_images=None):
        if all_images:
            if not mask_images or not gt_images:
                raise ValueError("Both mask_images and gt_images must be specified when all_images is True.")

            unique_classes = set()

            for mask_file, gt_file in zip(mask_images, gt_images):
                try:
                    # Load images
                    mask = cv2.imread(mask_file)
                    gt = cv2.imread(gt_file)

                    if mask is None:
                        raise ValueError(f"Failed to load mask image: {mask_file}")
                    if gt is None:
                        raise ValueError(f"Failed to load ground truth image: {gt_file}")

                    # Convert to RGB
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

                    # Validate dimensions
                    if mask.ndim != 3 or mask.shape[2] != 3:
                        raise ValueError(f"Mask image is not a valid RGB image: {mask_file}")
                    if gt.ndim != 3 or gt.shape[2] != 3:
                        raise ValueError(f"Ground truth image is not a valid RGB image: {gt_file}")

                    # Detect classes
                    classes = np.unique(np.concatenate((mask.reshape(-1, 3), gt.reshape(-1, 3))), axis=0)
                    for c in classes:
                        if len(c) == 3:  # Ensure valid RGB tuple
                            unique_classes.add(tuple(map(int, c)))  # Convert np.uint8 to Python int
                except Exception as e:
                    print(f"Error processing {mask_file} or {gt_file}: {e}")

            return sorted(unique_classes)

        else:
            try:
                mask = cv2.imread(mask_image)
                gt = cv2.imread(gt_image)

                if mask is None:
                    raise ValueError(f"Failed to load mask image: {mask_image}")
                if gt is None:
                    raise ValueError(f"Failed to load ground truth image: {gt_image}")

                # Convert to RGB
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

                # Validate dimensions
                if mask.ndim != 3 or mask.shape[2] != 3:
                    raise ValueError(f"Mask image is not a valid RGB image: {mask_image}")
                if gt.ndim != 3 or gt.shape[2] != 3:
                    raise ValueError(f"Ground truth image is not a valid RGB image: {gt_image}")

                # Detect classes
                classes = np.unique(np.concatenate((mask.reshape(-1, 3), gt.reshape(-1, 3))), axis=0)
                return [tuple(map(int, c)) for c in classes if not np.all(c == [0, 0, 0])]

            except Exception as e:
                print(f"Error processing {mask_image} or {gt_image}: {e}")

        return []
    
    def visualize_misclassification(self, pred_path, gt_path, target_class=None):
        # Load prediction and ground truth images as grayscale
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        print(f"Pred unique values: {np.unique(pred)}")
        print(f"GT unique values: {np.unique(gt)}")

        # Define colors for visualization
        colors = {
            "TP": [0, 255, 0],    # True Positive - Green
            "TN": [0, 0, 255],    # True Negative - Blue
            "FP": [255, 0, 0],    # False Positive - Red
            "FN": [255, 255, 0],  # False Negative - Yellow
            "NONE": [0, 0, 0]     # No visualization - Black
        }

        # Create an empty RGB image for visualization
        height, width = pred.shape[:2]
        visualization = np.zeros((height, width, 3), dtype=np.uint8)

        if target_class is not None:
            target_gray = int(0.299 * target_class[0] + 0.587 * target_class[1] + 0.114 * target_class[2])
            
            pred = (pred == target_gray).astype(np.uint8)
            gt = (gt == target_gray).astype(np.uint8)

        # Calculate misclassification categories
        tp = (pred == 1) & (gt == 1)  # True Positive
        tn = (pred == 0) & (gt == 0)  # True Negative
        fp = (pred == 1) & (gt == 0)  # False Positive
        fn = (pred == 0) & (gt == 1)  # False Negative

        # Debugging: Verify mask sizes and contents
        print(f"Target class RGB: {target_class}, Grayscale: {target_gray}")
        print(f"Pred unique values: {np.unique(pred)}")
        print(f"GT unique values: {np.unique(gt)}")

        print("TP pixels:", np.sum(tp))
        print("TN pixels:", np.sum(tn))
        print("FP pixels:", np.sum(fp))
        print("FN pixels:", np.sum(fn))
        print("Show TP:", self.legend.show_tp.get())
        print("Show TN:", self.legend.show_tn.get())
        print("Show FP:", self.legend.show_fp.get())
        print("Show FN:", self.legend.show_fn.get())

        # Assign colors to each category based on the legend's visibility
        if self.legend.show_tp.get():
            visualization[tp] = colors["TP"]
        if self.legend.show_tn.get():
            visualization[tn] = colors["TN"]
        if self.legend.show_fp.get():
            visualization[fp] = colors["FP"]
        if self.legend.show_fn.get():
            visualization[fn] = colors["FN"]

        return visualization