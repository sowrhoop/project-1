import os
import cv2
import logging
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image, ImageTk
from ultralytics import YOLO
from tqdm import tqdm  
import numpy as np
import gc  # Added garbage collection module
import torch
from torchvision.ops import box_iou
import hashlib
# Configure logging
logging.basicConfig(format="%(message)s", level=logging.INFO)
 
# Paths
folder_path = r"C:\Users\DELL\Desktop\AP_2000\AP_100"
temp_image_folder = r"temp_image"
poppler_path = r"D:\GANI\Release-24.08.0-0\poppler-24.08.0\Library\bin"
model_path = r"E:\eval3\eval run\best.pt"
feedback_file = r"feedback_testing_AP_100.xlsx"
no_feedback_folder = r"no_feedback_AP_100_testing"

 
# Ensure necessary folders exist
os.makedirs(temp_image_folder, exist_ok=True)
os.makedirs(no_feedback_folder, exist_ok=True)
 
# Load YOLO model
model = YOLO(model_path)
 
# Class labels
CLASS_LABELS = ['barcode', 'county_stamp', 'cross_out', 'handwriting', 'initial',
                'noise', 'notary_stamp', 'over_lapping', 'recording_stamp', 'signature', 'text' ,'tick']
 
# Classes to check for overlap with cross_out
OVERLAP_CLASSES = ['handwriting', 'initial', 'noise', 'over_lapping', 'signature', 'text' ,'tick']
# OVERLAP_CLASSES = ['handwriting', 'signature', 'tick']
def save_feedback(image_path,processed_image_path,response):
    """Saves user feedback in an Excel file."""
    try:
        feedback_data = pd.DataFrame([[os.path.basename(image_path), response]], columns=["Image", "Response"])
        if os.path.exists(feedback_file):
            existing_data = pd.read_excel(feedback_file)
            feedback_data = pd.concat([existing_data, feedback_data], ignore_index=True)
        feedback_data.to_excel(feedback_file, index=False)
        logging.info(f"Saved feedback for {os.path.basename(image_path)}: {response}")
        if response == "No":
            no_feedback_path = os.path.join(no_feedback_folder, os.path.basename(image_path))
            os.rename(image_path, no_feedback_path)
            logging.info(f"Moved image to no_feedback folder: {os.path.basename(image_path)}")
        no_feedback_path = os.path.join(no_feedback_folder, os.path.basename(processed_image_path))
        os.rename(image_path, no_feedback_path)
    except Exception as e:
        logging.error(f"Error saving feedback for {image_path}: {e}")
 
 
def filter_non_overlapping(cross_out_detections, other_detections, iou_threshold=0.25):
    """
    Filters out cross_out_detections that overlap more than the given IoU threshold with other_detections.
 
    Args:
        cross_out_detections (list): List of (bounding_box, class_label, confidence_score) tuples.
        other_detections (list): List of (bounding_box, class_label, confidence_score) tuples.
        iou_threshold (float): IoU threshold above which a detection is considered overlapping.
 
    Returns:
        list: Filtered list of cross_out_detections that do not overlap more than the threshold.
    """
    if not cross_out_detections or not other_detections:
        return cross_out_detections  # Return all if no other detections exist
   
    try:
        # Convert bounding boxes to PyTorch tensors
        cross_out_boxes = torch.tensor([det[0] for det in cross_out_detections], dtype=torch.float)
        other_boxes = torch.tensor([det[0] for det in other_detections], dtype=torch.float)
 
        # Compute IoU (Intersection over Union)
        iou_matrix = box_iou(cross_out_boxes, other_boxes)
 
        # Find indices of cross_out boxes that have IoU less than the threshold (0.25)
        non_overlapping_indices = (iou_matrix.max(dim=1).values < iou_threshold).nonzero(as_tuple=True)[0]
 
        # Keep only non-overlapping cross_out detections
        result = [cross_out_detections[i.item()] for i in non_overlapping_indices]
 
        # Clean up tensors to free memory
        del cross_out_boxes, other_boxes, iou_matrix, non_overlapping_indices
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
 
        return result
 
    except Exception as e:
        logging.error(f"Error in filter_non_overlapping: {e}")
        return cross_out_detections  # Fallback: return original list in case of failure
 
def convert_pdf_to_images(pdf_path):
    """Converts a PDF to images with memory optimization."""
    try:
        # Process one page at a time instead of loading all pages
        pdf_images = convert_from_path(pdf_path, poppler_path=poppler_path, thread_count=1)
        for i, image in enumerate(pdf_images):
            image_path = os.path.join(temp_image_folder, f"{os.path.basename(pdf_path)}_page_{i+1}.png")
            image.save(image_path, "PNG")
            yield image_path, i + 1
            # Clear image from memory
            del image
        # Clean up
        del pdf_images
        gc.collect()
    except Exception as e:
        logging.error(f"Error converting PDF {pdf_path}: {e}")
 
 
 
# Function to generate a consistent color per class
def get_class_color(class_name):
    """Generate a consistent unique color for a given class label."""
    hash_val = int(hashlib.md5(class_name.encode()).hexdigest(), 16)  # Hash class name
    np.random.seed(hash_val % (2**32))  # Seed randomness for consistency
    base_color = np.random.randint(50, 200, size=3).tolist()  # Avoid too dark/light colors
    return tuple(base_color)
 
def draw_bounding_boxes(image_path, detections):
    """Draws bounding boxes with class-based colors, labels, and red arrows to the nearest edge."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
 
        os.makedirs("temp_images", exist_ok=True)  # Ensure output folder exists
        class_colors = {}  # Store colors per class
 
        label_positions = []  # Track label positions to prevent overlap
        overlay = img.copy()  # For transparent filling
 
        for (box, detected_class, score) in detections:
            # Assign a consistent color to each class
            if detected_class not in class_colors:
                class_colors[detected_class] = get_class_color(detected_class)
           
            color = class_colors[detected_class]
            light_color = tuple(int(c * 0.6 + 100) for c in color)  # Generate light shade
 
            x1, y1, x2, y2 = map(int, box)
           
            # **Draw semi-transparent filled rectangle**
            cv2.rectangle(overlay, (x1, y1), (x2, y2), light_color, -1)  # Filled box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Border
 
            # **Prepare label text**
            label = f"{detected_class} ({score:.2f})"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
 
            # **Avoid label overlap**
            label_x, label_y = x2 + 10, y1 - 5  # Default: right of the box
            while any(abs(label_y - prev_y) < text_height + 10 for _, prev_y in label_positions):
                label_y += text_height + 10  
            label_positions.append((label_x, label_y))  
 
            # **Draw label background**
            cv2.rectangle(img, (label_x - 5, label_y - text_height - 5),
                          (label_x + text_width + 5, label_y + 5), color, -1)
 
            # **Adjust text color for visibility**
            brightness = np.mean(color)
            text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
 
            # **Draw label text**
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
 
            # **Find closest edge of the bounding box for arrow placement**
            box_center_x, box_center_y = (x1 + x2) // 2, (y1 + y2) // 2
            distances = {
                "left": abs(label_x - x1),
                "right": abs(label_x - x2),
                "top": abs(label_y - y1),
                "bottom": abs(label_y - y2),
            }
            closest_edge = min(distances, key=distances.get)
 
            if closest_edge == "left":
                arrow_x, arrow_y = x1, box_center_y
            elif closest_edge == "right":
                arrow_x, arrow_y = x2, box_center_y
            elif closest_edge == "top":
                arrow_x, arrow_y = box_center_x, y1
            else:  # bottom
                arrow_x, arrow_y = box_center_x, y2
 
            # **Draw red arrow from label to bounding box edge**
            cv2.arrowedLine(img, (label_x - 10, label_y - text_height // 2),
                            (arrow_x, arrow_y), (0, 0, 255), 2, tipLength=0.2)
 
        # **Blend overlay with transparency**
        alpha = 0.3  # Transparency factor for filled rectangles
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
 
        # **Save processed image**
        processed_image_path = os.path.join("temp_images", f"processed_{os.path.basename(image_path)}")
        success = cv2.imwrite(processed_image_path, img)
        if not success:
            raise ValueError(f"Failed to save processed image: {processed_image_path}")
 
        return processed_image_path
 
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
 
def show_image_with_buttons(image_path, image_name=None):
    """Displays the image with 'Yes' and 'No' buttons and zoom functionality."""
    if image_name is None:
        image_name = os.path.basename(image_path)
    result = [None]  # Use a list to store result since Python 3 closures capture by reference
    def on_yes():
        result[0] = "Yes"
        root.quit()
    def on_no():
        result[0] = "No"
        root.quit()
 
    # Initialize zoom level
    zoom_level = [1.0]
    font_size = [12]  # Starting font size for labels
    def zoom_in():
        zoom_level[0] *= 1.2
        font_size[0] = min(24, font_size[0] + 2)  # Increase font size but cap at 24
        update_image()
    def zoom_out():
        zoom_level[0] /= 1.2
        font_size[0] = max(8, font_size[0] - 2)  # Decrease font size but not below 8
        update_image()
    def update_image():
        try:
            # Load original image
            img = Image.open(image_path)
            width, height = img.size
            # Calculate new size based on zoom level
            new_width = int(width * zoom_level[0])
            new_height = int(height * zoom_level[0])
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            # Update image in display
            new_photo = ImageTk.PhotoImage(resized_img)
            panel.config(image=new_photo)
            panel.image = new_photo  # Keep a reference
            # Update zoom information
            zoom_label.config(text=f"Zoom: {zoom_level[0]:.2f}x", font=("Arial", font_size[0]))
            # Release resources
            del img, resized_img
        except Exception as e:
            logging.error(f"Error updating image: {e}")
    # Handle keyboard shortcuts
    def on_key_press(event):
        if event.char == '+' or event.keysym == 'plus' or event.keysym == 'equal':
            zoom_in()
        elif event.char == '-' or event.keysym == 'minus':
            zoom_out()
    # Setup main window
    root = tk.Tk()
    root.title(f"Review Crossout Detection - {image_name}")
    root.bind("<Key>", on_key_press)
    try:
        # Create main frame for image and controls
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        # Create frame for zoom controls
        zoom_frame = tk.Frame(main_frame)
        zoom_frame.pack(fill=tk.X)
        # Add zoom buttons
        zoom_in_btn = tk.Button(zoom_frame, text="Zoom In (+)", command=zoom_in)
        zoom_in_btn.pack(side=tk.LEFT, padx=5, pady=5)
        zoom_out_btn = tk.Button(zoom_frame, text="Zoom Out (-)", command=zoom_out)
        zoom_out_btn.pack(side=tk.LEFT, padx=5, pady=5)
        # Add zoom level indicator
        zoom_label = tk.Label(zoom_frame, text=f"Zoom: {zoom_level[0]:.2f}x", font=("Arial", font_size[0]))
        zoom_label.pack(side=tk.LEFT, padx=20, pady=5)
        # Instructions label
        instructions = tk.Label(zoom_frame, text="Use +/- keys or buttons to zoom", font=("Arial", 10))
        instructions.pack(side=tk.RIGHT, padx=10, pady=5)
        # Create scrollable frame for the image
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        # Add scrollbars
        h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # Create canvas for the image with scrollbars
        canvas = tk.Canvas(canvas_frame,
                           xscrollcommand=h_scrollbar.set,
                           yscrollcommand=v_scrollbar.set,
                           width=800, height=600)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Connect scrollbars to canvas
        h_scrollbar.config(command=canvas.xview)
        v_scrollbar.config(command=canvas.yview)
        # Create frame inside canvas to hold the image
        image_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=image_frame, anchor=tk.NW)
        # Load initial image
        img = Image.open(image_path)
        photo_img = ImageTk.PhotoImage(img)
        # Display image
        panel = tk.Label(image_frame, image=photo_img)
        panel.image = photo_img
        panel.pack()
        # Update canvas scroll region
        image_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox(tk.ALL))
        # Create frame for Yes/No buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        # Add Yes/No buttons
        yes_button = tk.Button(button_frame, text="Yes", command=on_yes, width=10, height=2, bg="green", fg="white")
        yes_button.pack(side=tk.LEFT, padx=10, pady=10, expand=True)
        no_button = tk.Button(button_frame, text="No", command=on_no, width=10, height=2, bg="red", fg="white")
        no_button.pack(side=tk.RIGHT, padx=10, pady=10, expand=True)
        # Center the window
        root.update_idletasks()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = min(img.width + 50, screen_width - 100)
        window_height = min(img.height + 200, screen_height - 100)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        # Start main loop
        root.mainloop()
        # Clean up resources
        del img, photo_img
        root.destroy()
        return result[0]
    except Exception as e:
        logging.error(f"Error showing image {image_path}: {e}")
        try:
            root.destroy()
        except:
            pass
        return "No"
 
def process_pdf(pdf_file):
    pdf_path = os.path.join(folder_path, pdf_file)
    logging.info(f"Processing PDF: {pdf_file}")
    try:
        for image_path, page_num in convert_pdf_to_images(pdf_path):
            try:
                # Run model with lower precision
                with torch.no_grad():  # Disable gradient calculation
                    results = model(image_path)
                # Extract detections efficiently
                detections = []
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    for box, score, cls in zip(boxes, scores, classes):
                        detections.append((box, CLASS_LABELS[cls], score))
                # Filter by class more efficiently
                cross_out_detections = []
                other_detections = []
                for det in detections:
                    if det[1] == "cross_out":
                        cross_out_detections.append(det)
                    elif det[1] in OVERLAP_CLASSES:
                        other_detections.append(det)
                # Free memory
                del results, boxes, scores, classes
                gc.collect()
                # Only process if we found cross_out detections
                if cross_out_detections:
                    filtered_cross_outs = filter_non_overlapping(cross_out_detections, other_detections)
                    if filtered_cross_outs:
                        processed_image_path = draw_bounding_boxes(image_path, cross_out_detections+other_detections)
                        # Get feedback from user
                        feedback = show_image_with_buttons(processed_image_path, f"{pdf_file}_page_{page_num}")
                        # Make sure feedback is properly saved
                        if feedback:  # Check if feedback is not None
                            save_feedback(image_path,processed_image_path, feedback)
                        else:
                            logging.warning(f"No feedback received for {processed_image_path}")
                           
                # Remove temporary image file if it exists
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                logging.error(f"Error processing page {page_num} of {pdf_file}: {e}")
                continue
            # Force garbage collection after each page
            gc.collect()
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_file}: {e}")
 
if __name__ == '__main__':
    pdf_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".pdf"))
    # Process files in batches to manage memory
    batch_size = 5  # Adjust based on your system's memory
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i+batch_size]
        for pdf_file in tqdm(batch, desc=f"Processing PDFs (batch {i//batch_size + 1})"):
            process_pdf(pdf_file)
            gc.collect()  # Force garbage collection after each PDF
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
 
 
 
 
 
 
 
 
# ... rest of the existing code ...