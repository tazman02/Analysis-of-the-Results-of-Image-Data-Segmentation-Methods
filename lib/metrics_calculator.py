# MetricsCalculator is a class designed to compute and display various evaluation metrics 
#     for image segmentation tasks. It provides functionality to calculate metrics such as 
#     Dice coefficient, Intersection over Union (IoU), accuracy, precision, recall, specificity, 
#     fallout, false negative rate (FNR), volumetric similarity, area under the curve (AUC), 
#     generalized consistency error (GCE), Cohen's kappa, average Hausdorff distance (AHD), 
#     average symmetric surface distance (ASSD), surface Dice similarity coefficient (DSC), 
#     boundary IoU, and average precision (AP).
#     Attributes:
#         stats_frame (tk.Frame): A frame to hold the statistics table.
#         class_manager (object): Manages class-related operations.
#         analyzer (object): Analyzes the input data.
#         app (object): Reference to the main application.
#         table (ttk.Treeview): A Treeview widget to display per-image metrics.
#         table_stats (ttk.Treeview): A Treeview widget to display dataset-wide metrics.
#         export (ExportMetrics): Handles exporting metrics to external files.
#     Methods:
#         compute_dice(tp, tn, fn, fp): Computes the Dice coefficient.
#         compute_iou(tp, tn, fn, fp): Computes the Intersection over Union (IoU).
#         compute_accuracy(tp, tn, fn, fp): Computes the accuracy.
#         compute_precision(tp, tn, fn, fp): Computes the precision.
#         compute_recall(tp, tn, fn, fp): Computes the recall.
#         compute_specificity(tp, tn, fn, fp): Computes the specificity.
#         compute_fallout(tp, tn, fn, fp): Computes the fallout.
#         compute_fnr(tp, tn, fn, fp): Computes the false negative rate (FNR).
#         compute_vol_similarity(tp, tn, fn, fp): Computes the volumetric similarity.
#         compute_auc(fallout, fnr): Computes the area under the curve (AUC).
#         compute_metrics(pred, gt, target_class): Computes all metrics for a given prediction and ground truth.
#         compute_assd(gt_mask, pred_mask): Computes the Average Symmetric Surface Distance (ASSD).
#         compute_ap(recall, precision): Computes the Average Precision (AP).
#         compute_surface_dsc(gt_mask, pred_mask, tau): Computes the Surface Dice Similarity Coefficient (DSC).
#         compute_average_hausdorff_distance(maskA, maskB): Computes the Average Hausdorff Distance (AHD).
#         compute_gce(pred, gt, tp, tn, fp, fn): Computes the Generalized Consistency Error (GCE).
#         boundary_iou(mask1, mask2, dilation_ratio): Computes the Boundary IoU.
#         compute_kappa(tp, tn, fp, fn): Computes Cohen's kappa.
#         calculate_and_display_metrics(mask_images, gt_images): Calculates metrics for a set of images and updates the UI.
#         process_queue(): Processes the metrics queue and updates the UI.
#         compute_metrics_from_totals(tp, tn, fp, fn, pred, gt, total_recall, total_precision): 
#             Computes aggregate metrics from totals across all images.
import cv2
import numpy as np
import tkinter as tk
import threading
import queue
from tkinter.filedialog import asksaveasfilename
from tkinter import ttk
from skimage.segmentation import find_boundaries
from scipy.spatial import KDTree

from lib.export_metrics import ExportMetrics

ROUND_DIGITS = 4

class MetricsCalculator:
    def __init__(self, root, class_manager, analyzer, app, stats_frame):
        # Create a Frame to hold the table and scrollbars
        frame = tk.Frame(root)
        frame.pack(fill="both", expand=True)
        self.stats_frame = stats_frame

        self.class_manager = class_manager
        self.analyzer = analyzer
        self.app = app

        # Create the Treeview widget
        self.table = ttk.Treeview(
            frame,
            columns=("Image", "Class", "Dice", "IoU", "Accuracy", "Precision", "Recall", "Specificity", "Fallout", "FNR", "Volumetric similarity", "AUC", "GCE score", "Kappa", "AHD", "ASSD", "DSC", "Boundary IoU", "AP"),
            show="headings",
            height=20
        )
        
        # Add the vertical scrollbar
        y_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=y_scrollbar.set)

        # Add the horizontal scrollbar
        x_scrollbar = ttk.Scrollbar(frame, orient="horizontal", command=self.table.xview)
        self.table.configure(xscrollcommand=x_scrollbar.set)

        # Use grid for better layout control
        self.table.grid(row=0, column=0, sticky="nsew")
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        x_scrollbar.grid(row=1, column=0, sticky="ew")

        # Configure frame resizing behavior
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)


        for col in self.table["columns"]:
            self.table.heading(col, text=col)
            self.table.column(col, width=100, anchor="center", stretch=False)

        frame_stats = tk.Frame(self.stats_frame)
        frame_stats.pack(fill="both", expand=True)
        
        self.table_stats = ttk.Treeview(
        frame_stats,
        columns=("Name", "Class", "Dice", "IoU", "Accuracy", "Precision", "Recall", "Specificity", "Fallout", "FNR", "Volumetric similarity", "AUC", "GCE score", "Kappa", "AHD", "ASSD", "DSC", "Boundary IoU", "AP"),
        show="headings",
        height=10 
        )
        
        y_scrollbar_stats = ttk.Scrollbar(frame_stats, orient="vertical", command=self.table_stats.yview)
        self.table_stats.configure(yscrollcommand=y_scrollbar_stats.set)
        
        x_scrollbar_stats = ttk.Scrollbar(frame_stats, orient="horizontal", command=self.table_stats.xview)
        self.table_stats.configure(xscrollcommand=x_scrollbar_stats.set)
        
        self.table_stats.grid(row=0, column=0, sticky="nsew")
        y_scrollbar_stats.grid(row=0, column=1, sticky="ns")
        x_scrollbar_stats.grid(row=1, column=0, sticky="ew")
        
        frame_stats.grid_rowconfigure(0, weight=1)
        frame_stats.grid_columnconfigure(0, weight=1)
        
        for col in self.table_stats["columns"]:
            self.table_stats.heading(col, text=col)


        self.export = ExportMetrics(self.table, self.app)
    def compute_dice(self, tp, tn, fn, fp):
        return float((2.0 * tp) / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0
    def compute_iou(self, tp, tn, fn, fp):
        return float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
    def compute_accuracy(self, tp, tn, fn, fp):
        return float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0
    def compute_precision(self, tp, tn, fn, fp):
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    def compute_recall(self, tp, tn, fn, fp):
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    def compute_specificity(self, tp, tn, fn, fp):
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    def compute_fallout(self, tp, tn, fn, fp):
        return float(fp / (fp + tn)) if (tn + fp) > 0 else 0.0
    def compute_fnr(self, tp, tn, fn, fp):
        return float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    def compute_vol_similarity(self, tp, tn, fn, fp):
        return float(1 - (abs(fn - fp)/(2*tp + fp + fn))) if (2*tp + fp + fn) > 0 else 0.0
    def compute_auc(self, fallout, fnr):
        return float(1 - (1/2) * (fallout + fnr))
        
    def compute_metrics(self, pred, gt, target_class):
        
        target_class = int(0.299 * target_class[0] + 0.587 * target_class[1] + 0.114 * target_class[2])
        
        tp = np.sum((pred == target_class) & (gt == target_class))
        tn = np.sum((pred != target_class) & (gt != target_class))
        fp = np.sum((pred == target_class) & (gt != target_class))
        fn = np.sum((pred != target_class) & (gt == target_class))

        dice = round(self.compute_dice(tp, tn, fn, fp), ROUND_DIGITS)
        iou = round(self.compute_iou(tp, tn, fn, fp), ROUND_DIGITS)
        accuracy = round(self.compute_accuracy(tp, tn, fn, fp), ROUND_DIGITS)
        precision = round(self.compute_precision(tp, tn, fn, fp), ROUND_DIGITS)
        recall = round(self.compute_recall(tp, tn, fn, fp), ROUND_DIGITS)
        specificity = round(self.compute_specificity(tp, tn, fn, fp), ROUND_DIGITS)
        fallout = round(self.compute_fallout(tp, tn, fn, fp), ROUND_DIGITS)
        fnr = round(self.compute_fnr(tp, tn, fn, fp), ROUND_DIGITS)
        vol_similarity = round(self.compute_vol_similarity(tp, tn, fn, fp), ROUND_DIGITS)
        auc = round(self.compute_auc(fallout, fnr), ROUND_DIGITS)
        
        binary_mask_pred = (pred == target_class).astype(np.uint8)
        binary_mask_gt = (gt == target_class).astype(np.uint8)

        boundary_iou_score = round(self.boundary_iou(binary_mask_pred, binary_mask_gt), ROUND_DIGITS)
        #gce
        gce_score = round(self.compute_gce(pred, gt, tp, tn, fp, fn) , ROUND_DIGITS)
        
        #kappa   
        kappa_score = round(self.compute_kappa(tp, tn, fp, fn), ROUND_DIGITS)
        
        #hausdorf dist
        ahd_score = round(self.compute_average_hausdorff_distance(binary_mask_pred, binary_mask_gt), ROUND_DIGITS)
        
        #assd
        assd_score = round(self.compute_assd(binary_mask_gt, binary_mask_pred), ROUND_DIGITS)
        
        #ap
        #surface DSC
        dsc_score = round(self.compute_surface_dsc(binary_mask_gt, binary_mask_pred), ROUND_DIGITS)
        
        return {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "dice": dice,
            "iou": iou,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "fallout": fallout,
            "fnr": fnr,
            "vol_similarity": vol_similarity,
            "gce_score": gce_score,
            "kappa_score": kappa_score,
            "ahd_score": ahd_score,
            "assd_score": assd_score,
            "dsc_score": dsc_score,
            "boundary_iou_score": boundary_iou_score,
            "auc": auc
        }
        
        
    def compute_assd(self, gt_mask, pred_mask):
        """
        Computes the Average Symmetric Surface Distance (ASSD).
        
        Args:
            gt_mask (numpy.ndarray): Ground truth segmentation mask.
            pred_mask (numpy.ndarray): Predicted segmentation mask.
        
        Returns:
            float: Average Symmetric Surface Distance (ASSD).
        """
        def extract_boundary(mask):
            boundary = find_boundaries(mask, mode='inner').astype(np.uint8)
            return np.column_stack(np.where(boundary > 0))
        # Extract boundary points
        
        gt_boundary = extract_boundary(gt_mask)
        pred_boundary = extract_boundary(pred_mask)
        
        if len(gt_boundary) == 0 or len(pred_boundary) == 0:
            return float('inf')  # If no boundary, return infinity

        # Compute nearest neighbor distances using KDTree
        gt_tree = KDTree(gt_boundary)
        pred_tree = KDTree(pred_boundary)

        # Compute distances
        d_gt_to_pred, _ = pred_tree.query(gt_boundary)  # GT to Prediction
        d_pred_to_gt, _ = gt_tree.query(pred_boundary)  # Prediction to GT

        # Compute the mean distance (ASSD)
        assd = ((np.mean(d_gt_to_pred) + np.mean(d_pred_to_gt)) / 2.0) 

        return assd
        
    def compute_ap(self, recall, precision):
        """
        Compute Average Precision (AP) using trapezoidal integration.

        Args:
            recall (list or np.ndarray): Recall values (0 to 1).
            precision (list or np.ndarray): Precision values (0 to 1).

        Returns:
            float: Average Precision (AP).
        """
        recall = np.array(recall)
        precision = np.array(precision)

        # Sort recall and precision if needed
        if not np.all(np.diff(recall) >= 0):
            sorted_indices = np.argsort(recall)
            recall = recall[sorted_indices]
            precision = precision[sorted_indices]

        # Ensure full coverage of recall [0, 1]
        if recall[0] > 0:
            recall = np.insert(recall, 0, 0.0)
            precision = np.insert(precision, 0, precision[0])
        if recall[-1] < 1:
            recall = np.append(recall, 1.0)
            precision = np.append(precision, 0.0)

        # Use trapezoidal integration
        ap = np.trapz(precision, recall)
        return ap


    def compute_surface_dsc(self, gt_mask, pred_mask, tau=2):
        """
        Computes the Surface Dice Similarity Coefficient (Surface DSC).
        
        Args:
            gt_mask (numpy.ndarray): Ground truth binary mask (0 or 255).
            pred_mask (numpy.ndarray): Predicted binary mask (0 or 255).
            tau (int): Tolerance in pixels for surface comparison.
        
        Returns:
            float: Surface Dice score.
        """
        
        def extract_boundary(mask):
            kernel = np.ones((3, 3), dtype=np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            eroded = cv2.erode(mask, kernel, iterations=1)
            boundary = cv2.absdiff(dilated, eroded)
            return np.column_stack(np.where(boundary > 0))
        
        gt_boundary = extract_boundary(gt_mask)
        pred_boundary = extract_boundary(pred_mask)
        
        if len(gt_boundary) == 0 and len(pred_boundary) == 0:
            return 1.0  # Perfect match (both empty)
        elif len(gt_boundary) == 0 or len(pred_boundary) == 0:
            return 0.0  # One is empty

        gt_tree = KDTree(gt_boundary)
        pred_tree = KDTree(pred_boundary)

        d_gt_to_pred, _ = pred_tree.query(gt_boundary)
        d_pred_to_gt, _ = gt_tree.query(pred_boundary)

        TP = np.sum(d_gt_to_pred <= tau)
        FN = np.sum(d_gt_to_pred > tau)
        FP = np.sum(d_pred_to_gt > tau)

        surface_dsc = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
        return surface_dsc

    def compute_average_hausdorff_distance(self, maskA, maskB):
        """
        Compute the Average Hausdorff Distance (AHD) between two binary masks.
        The Average Hausdorff Distance is a measure of similarity between two sets of points.
        It calculates the average of the mean distances from points in one set to the closest 
        points in the other set, and vice versa.
        Args:
            maskA (numpy.ndarray): A binary mask representing the first set of points.
            maskB (numpy.ndarray): A binary mask representing the second set of points.
        Returns:
            float: The Average Hausdorff Distance between the two masks. If either mask 
                   does not contain any points, returns `float('inf')` to indicate a missing boundary.
        Notes:
            - The function uses KDTree for efficient nearest-neighbor queries.
            - If either mask is empty, the function returns a large value (`float('inf')`) 
              to signify the absence of a boundary.
        """
        def extract_points_from_mask(mask):
            points = np.column_stack(np.where(mask > 0))  # Extract pixel coordinates
            return points

        setA = extract_points_from_mask(maskA)
        setB = extract_points_from_mask(maskB)

        # Check if either set is empty
        if len(setA) == 0 or len(setB) == 0:
            return float('inf')  # Large value to indicate missing boundary

        # Build KDTree
        treeA = KDTree(setA)
        treeB = KDTree(setB)

        # Compute distances
        distancesA = treeB.query(setA)[0] if len(setB) > 0 else np.array([])
        distancesB = treeA.query(setB)[0] if len(setA) > 0 else np.array([])

        # Handle empty mean computation
        meanA = np.mean(distancesA) if len(distancesA) > 0 else float('inf')
        meanB = np.mean(distancesB) if len(distancesB) > 0 else float('inf')

        ahd = (meanA + meanB) / 2.0
        return ahd
            
    def compute_gce(self, pred, gt, tp, tn, fp, fn):
        """
        Compute the Global Consistency Error (GCE) between predicted and ground truth segmentations.
        The GCE measures the degree of overlap between two segmentations, considering both over-segmentation
        and under-segmentation errors. It is calculated as the minimum of two directional errors.
        Args:
            pred (numpy.ndarray): The predicted segmentation mask, a 2D array.
            gt (numpy.ndarray): The ground truth segmentation mask, a 2D array.
            tp (int): The number of true positive pixels.
            tn (int): The number of true negative pixels.
            fp (int): The number of false positive pixels.
            fn (int): The number of false negative pixels.
        Returns:
            float: The computed GCE value. A lower value indicates better segmentation consistency.
        """
        
        if pred.shape == gt.shape:
            n = pred.shape[0] * pred.shape[1]
            
            gce = (1 / n) * min(
                (fn * (fn + 2 * tp) / (tp + fn) if (tp + fn) != 0 else 0) +
                (fp * (fp + 2 * tn) / (tn + fp) if (tn + fp) != 0 else 0),
                
                (fp * (fp + 2 * tp) / (tp + fp) if (tp + fp) != 0 else 0) +
                (fn * (fn + 2 * tn) / (tn + fn) if (tn + fn) != 0 else 0)
            ) if n != 0 else 0

            
            return gce
    

    def boundary_iou(self, mask1, mask2, dilation_ratio=0.1):
        """
        Compute Boundary IoU (BIoU) between two binary masks.
        
        Args:
            mask1 (np.ndarray): First binary mask (H, W), values 0 or 255.
            mask2 (np.ndarray): Second binary mask (H, W), values 0 or 255.
            dilation_ratio (float): Relative boundary thickness (w.r.t. image size).
        
        Returns:
            float: Boundary IoU score.
        """
        def get_boundary(mask, dilation_ratio):
            h, w = mask.shape
            dilation = max(1, int(round(dilation_ratio * np.sqrt(h * w))))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(mask, kernel, iterations=dilation)
            eroded = cv2.erode(mask, kernel, iterations=dilation)
            boundary = cv2.absdiff(dilated, eroded)
            return boundary

        # Ensure binary masks (0 or 255)
        mask1 = np.where(mask1 > 0, 255, 0).astype(np.uint8)
        mask2 = np.where(mask2 > 0, 255, 0).astype(np.uint8)

        boundary1 = get_boundary(mask1, dilation_ratio)
        boundary2 = get_boundary(mask2, dilation_ratio)

        intersection = np.logical_and(boundary1 > 0, boundary2 > 0).sum()
        union = np.logical_or(boundary1 > 0, boundary2 > 0).sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return intersection / union
    
    def compute_kappa(self, tp, tn, fp, fn):
        """
        Compute Cohen's kappa coefficient.
        Cohen's kappa is a statistical measure of inter-rater agreement or reliability.
        It is generally used to evaluate the agreement between two raters or classifiers
        on a classification task.
        Parameters:
            tp (int): True positives - the number of correctly predicted positive instances.
            tn (int): True negatives - the number of correctly predicted negative instances.
            fp (int): False positives - the number of negative instances incorrectly predicted as positive.
            fn (int): False negatives - the number of positive instances incorrectly predicted as negative.
        Returns:
            float: The computed Cohen's kappa coefficient, a value between -1 and 1.
                   A value of 1 indicates perfect agreement, 0 indicates no agreement
                   beyond chance, and -1 indicates complete disagreement.
        """
        f = ((tn + fn)*(tn + fp)+(fp + tp)*(fn + tp))/(tp + tn + fn + fp)
        kappa = ((tp + tn) - f)/((tp + tn + fn + fp) - f)
        
        return kappa

    def calculate_and_display_metrics(self, mask_images, gt_images):
        """
        Calculate and display evaluation metrics for predicted and ground truth images.
        This function computes various evaluation metrics for each pair of predicted 
        and ground truth images, as well as dataset-wide metrics. The results are 
        displayed in a thread-safe manner in the application's UI tables.
        Args:
            mask_images (list of str): List of file paths to the predicted mask images.
            gt_images (list of str): List of file paths to the ground truth images.
        Functionality:
            - Loads predicted and ground truth images into memory.
            - Computes per-class metrics for each image, including Dice, IoU, accuracy, 
              precision, recall, specificity, and others.
            - Computes macro and micro averages for each image.
            - Aggregates dataset-wide metrics, including per-class macro averages and 
              overall micro averages.
            - Updates the UI tables with the computed metrics in a thread-safe manner.
        Notes:
            - The computation is performed in a separate thread to prevent UI freezing.
            - The function ensures that UI updates are executed on the main thread.
            - Background class (0, 0, 0) is skipped during metric computation.
        Raises:
            None
        """
        if self.analysis_done:
            return
        self.analysis_done = True

        self.table.delete(*self.table.get_children())  # Clear previous data
        self.table_stats.delete(*self.table_stats.get_children())  # Clear dataset-wide table

        # Queue for thread-safe UI updates
        self.metrics_queue = queue.Queue()

        def compute():
            """Threaded computation to prevent UI freezing."""
            try:
                per_class_metrics_set = {}
                total_total_tp = total_total_tn = total_total_fp = total_total_fn = 0

                pred_images_loaded = []
                gt_images_loaded = []

                # Load images into memory
                for pred_path, gt_path in zip(mask_images, gt_images):
                    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    pred_images_loaded.append(pred)
                    gt_images_loaded.append(gt)

                # Compute metrics for each image
                for i, (pred, gt, pred_path, gt_path) in enumerate(zip(pred_images_loaded, gt_images_loaded, mask_images, gt_images)):
                    total_recall = []
                    total_precision = []
                    unique_classes = self.analyzer.detect_classes(pred_path, gt_path, False, None, None)
                    per_class_metrics = []
                    total_tp = total_tn = total_fp = total_fn = 0

                    # Process each class in the image
                    for target_class in unique_classes:
                        if target_class == (0, 0, 0):  # Skip background class
                            continue

                        metrics = self.compute_metrics(pred, gt, target_class)
                        per_class_metrics.append(metrics)

                        if target_class in per_class_metrics_set:
                            per_class_metrics_set[target_class] = {
                                key: (np.mean([per_class_metrics_set[target_class][key], metrics[key]]))
                                for key in metrics.keys() if isinstance(metrics[key], (float, int))
                            }
                        else:
                            per_class_metrics_set[target_class] = metrics

                        total_tp += metrics["tp"]
                        total_tn += metrics["tn"]
                        total_fp += metrics["fp"]
                        total_fn += metrics["fn"]

                        total_total_tp += metrics["tp"]
                        total_total_tn += metrics["tn"]
                        total_total_fp += metrics["fp"]
                        total_total_fn += metrics["fn"]
                        total_recall.append(metrics["recall"])
                        total_precision.append(metrics["precision"])

                        # Add class-level metrics to **BOTH** tables
                        row_data = (
                            f"Image {i + 1}",
                            f"Class {self.class_manager.get_class_name(target_class)}",
                            metrics["dice"], metrics["iou"], metrics["accuracy"], metrics["precision"],
                            metrics["recall"], metrics["specificity"], metrics["fallout"], metrics["fnr"],
                            metrics["vol_similarity"], metrics["auc"], metrics["gce_score"], metrics["kappa_score"],
                            metrics["ahd_score"], metrics["assd_score"], metrics["dsc_score"],
                            metrics["boundary_iou_score"], "-"
                        )
                        self.metrics_queue.put(("table", row_data))

                    # Compute Macro Metrics
                    macro_metrics = {
                        key: np.mean([m[key] for m in per_class_metrics if key in m])
                        for key in per_class_metrics[0].keys() if isinstance(per_class_metrics[0][key], (float, int))
                    }

                    # Compute Micro Metrics
                    micro_metrics = self.compute_metrics_from_totals(total_tp, total_tn, total_fp, total_fn, pred, gt, total_recall, total_precision)

                    # Add macro and micro metrics to **BOTH** tables
                    self.metrics_queue.put(("table", (
                        f"Image {i + 1}",
                        "Macro Avg",
                        macro_metrics["dice"], macro_metrics["iou"], macro_metrics["accuracy"],
                        macro_metrics["precision"], macro_metrics["recall"], macro_metrics["specificity"],
                        macro_metrics["fallout"], macro_metrics["fnr"], macro_metrics["vol_similarity"],
                        "-", macro_metrics["gce_score"], macro_metrics["kappa_score"],
                        macro_metrics["ahd_score"], macro_metrics["assd_score"], macro_metrics["dsc_score"],
                        macro_metrics["boundary_iou_score"], "-"
                    )))

                    self.metrics_queue.put(("table", (
                        f"Image {i + 1}",
                        "Micro Avg",
                        micro_metrics["dice"], micro_metrics["iou"], micro_metrics["accuracy"],
                        micro_metrics["precision"], micro_metrics["recall"], micro_metrics["specificity"],
                        micro_metrics["fallout"], micro_metrics["fnr"], micro_metrics["vol_similarity"],
                        micro_metrics["auc"], micro_metrics["gce_score"], micro_metrics["kappa_score"],
                        "-", "-", "-", "-", micro_metrics["ap_score"]
                    )))

                # Dataset-level per-class summary
                for class_label in sorted(per_class_metrics_set.keys()):
                    self.metrics_queue.put(("table_stats", (
                        f"Dataset class {class_label}",
                        "Macro Avg",
                        per_class_metrics_set[class_label]["dice"], per_class_metrics_set[class_label]["iou"],
                        per_class_metrics_set[class_label]["accuracy"], per_class_metrics_set[class_label]["precision"],
                        per_class_metrics_set[class_label]["recall"], per_class_metrics_set[class_label]["specificity"],
                        per_class_metrics_set[class_label]["fallout"], per_class_metrics_set[class_label]["fnr"],
                        per_class_metrics_set[class_label]["vol_similarity"], "-",
                        per_class_metrics_set[class_label]["gce_score"], per_class_metrics_set[class_label]["kappa_score"],
                        per_class_metrics_set[class_label]["ahd_score"], per_class_metrics_set[class_label]["assd_score"],
                        per_class_metrics_set[class_label]["dsc_score"], per_class_metrics_set[class_label]["boundary_iou_score"],
                        "-"
                    )))

                # Compute Overall Dataset Micro Metrics
                dataset_micro_metrics = self.compute_metrics_from_totals(
                    total_total_tp, total_total_tn, total_total_fp, total_total_fn, pred, gt, total_recall, total_precision
                )

                self.metrics_queue.put(("table_stats", (
                    "Dataset",
                    "Micro Avg",
                    dataset_micro_metrics["dice"], dataset_micro_metrics["iou"], dataset_micro_metrics["accuracy"],
                    dataset_micro_metrics["precision"], dataset_micro_metrics["recall"], dataset_micro_metrics["specificity"],
                    dataset_micro_metrics["fallout"], dataset_micro_metrics["fnr"], dataset_micro_metrics["vol_similarity"],
                    dataset_micro_metrics["auc"], dataset_micro_metrics["gce_score"], dataset_micro_metrics["kappa_score"],
                    "-", "-", "-", "-", dataset_micro_metrics["ap_score"]
                )))

            finally:
                self.app.root.after(10, self.process_queue)

        # Run computation in a separate thread
        threading.Thread(target=compute, daemon=True).start()

    def process_queue(self):
        """Processes the queue and updates the UI in bulk to avoid missing data."""
        while not self.metrics_queue.empty():
            table_name, row_data = self.metrics_queue.get()
            
            if table_name == "table":
                self.table.insert("", "end", values=row_data)
            elif table_name == "table_stats":
                self.table_stats.insert("", "end", values=row_data)

        # Close the loading window if calculations are done
        self.app.close_loading_window()

    def compute_metrics_from_totals(self, tp, tn, fp, fn, pred, gt, total_recall, total_precision):
        """
        Compute various evaluation metrics from the given totals and predictions.
        This function calculates a comprehensive set of metrics to evaluate the 
        performance of a classification or segmentation model. The metrics include 
        Dice coefficient, Intersection over Union (IoU), accuracy, precision, recall, 
        specificity, fallout, false negative rate (FNR), volumetric similarity, 
        area under the curve (AUC), global consistency error (GCE), Cohen's kappa, 
        and average precision (AP).
        Args:
            tp (int): True positives.
            tn (int): True negatives.
            fp (int): False positives.
            fn (int): False negatives.
            pred (array-like): Predicted labels or probabilities.
            gt (array-like): Ground truth labels.
            total_recall (float): Total recall value for AP calculation.
            total_precision (float): Total precision value for AP calculation.
        Returns:
            dict: A dictionary containing the computed metrics:
                - "dice" (float): Dice coefficient.
                - "iou" (float): Intersection over Union.
                - "accuracy" (float): Accuracy.
                - "precision" (float): Precision.
                - "recall" (float): Recall.
                - "specificity" (float): Specificity.
                - "fallout" (float): Fallout.
                - "fnr" (float): False negative rate.
                - "vol_similarity" (float): Volumetric similarity.
                - "auc" (float): Area under the curve.
                - "gce_score" (float): Global consistency error score.
                - "kappa_score" (float): Cohen's kappa score.
                - "ap_score" (float): Average precision score.
        """
        
        dice = round(self.compute_dice(tp, tn, fn, fp), ROUND_DIGITS)
        iou = round(self.compute_iou(tp, tn, fn, fp), ROUND_DIGITS)
        accuracy = round(self.compute_accuracy(tp, tn, fn, fp), ROUND_DIGITS)
        precision = round(self.compute_precision(tp, tn, fn, fp), ROUND_DIGITS)
        recall = round(self.compute_recall(tp, tn, fn, fp), ROUND_DIGITS)
        specificity = round(self.compute_specificity(tp, tn, fn, fp), ROUND_DIGITS)
        fallout = round(self.compute_fallout(tp, tn, fn, fp), ROUND_DIGITS)
        fnr = round(self.compute_fnr(tp, tn, fn, fp), ROUND_DIGITS)
        vol_similarity = round(self.compute_vol_similarity(tp, tn, fn, fp), ROUND_DIGITS)
        auc = round(self.compute_auc(fallout, fnr), ROUND_DIGITS)
                #gce
        gce_score = round(self.compute_gce(pred, gt, tp, tn, fp, fn) , ROUND_DIGITS)
        
        #kappa   
        kappa_score = round(self.compute_kappa(tp, tn, fp, fn), ROUND_DIGITS)
        
        #ap
        ap_score = round(self.compute_ap(total_recall, total_precision), ROUND_DIGITS)

        return {
            "dice": dice,
            "iou": iou,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "fallout": fallout,
            "fnr": fnr,
            "vol_similarity": vol_similarity,
            "auc": auc,
            "gce_score": gce_score,
            "kappa_score": kappa_score,
            "ap_score": ap_score
        }