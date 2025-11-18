import sys
import os
import glob

# Try to import heavy libraries, fail gracefully
try:
    import cv2
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    _SAM_LOADED = True
except ImportError as e:
    print(f"Warning: Failed to import AI libraries. {e}")
    print("The 'Identify Worms' feature will be disabled.")
    _SAM_LOADED = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QStackedWidget,
    QSizePolicy, QMessageBox
)
from PyQt5.QtGui import QPixmap, QFont, QPainter, QBrush, QColor, QImage
from PyQt5.QtCore import Qt, QSize

class MainWindow(QMainWindow):
    """Main application window that holds the stacked widget for pages."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Worm Image Processor")
        self.setGeometry(100, 100, 800, 600)

        # Create the stacked widget
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Store image files
        self.image_files = []

        # Create pages
        self.upload_page = UploadWidget(self)
        self.viewer_page = ViewerWidget(self)
        self.results_page = ResultsWidget(self)

        # Add pages to the stack
        self.stacked_widget.addWidget(self.upload_page)    # index 0
        self.stacked_widget.addWidget(self.viewer_page)    # index 1
        self.stacked_widget.addWidget(self.results_page)   # index 2

        # Start on the upload page
        self.show_upload_page()

    def show_upload_page(self):
        """Switches to the upload page (index 0)."""
        self.stacked_widget.setCurrentIndex(0)

    def show_viewer_page(self, files):
        """Switches to the viewer page (index 1) and passes image files."""
        if files:
            self.image_files = files
            self.viewer_page.set_image_files(self.image_files)
            self.stacked_widget.setCurrentIndex(1)

    def show_results_page(self, from_button):
        """
        Switches to the results page (index 2) and shows which
        button was clicked.
        """
        self.results_page.update_title(from_button)
        self.stacked_widget.setCurrentIndex(2)

class UploadWidget(QWidget):
    """Page 1: For selecting the image folder."""
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

        # Title Label
        title_label = QLabel("Select Image Folder")
        title_font = title_label.font()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Info Label
        info_label = QLabel(
            "Please choose a folder containing images named\n"
            "'0.jpg', '1.jpg', '2.jpg', etc."
        )
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #555;")
        layout.addWidget(info_label)

        # Upload Button
        self.upload_button = QPushButton("Select Folder")
        self.upload_button.setMinimumSize(200, 50)
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #007BFF;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        self.upload_button.clicked.connect(self.open_folder_dialog)
        
        # Add stretch to center the button
        layout.addStretch(1)
        layout.addWidget(self.upload_button, 0, Qt.AlignCenter)
        layout.addStretch(2)


    def open_folder_dialog(self):
        """Opens a dialog to select a directory."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            # Find all .jpg files
            image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
            
            # Sort files numerically based on their name (e.g., "0.jpg", "1.jpg")
            sorted_files = []
            for f in image_paths:
                basename = os.path.basename(f)
                filename, _ = os.path.splitext(basename)
                try:
                    # Store (integer_name, full_path)
                    sorted_files.append((int(filename), f))
                except ValueError:
                    # Ignore files that are not named with an integer
                    print(f"Skipping non-numeric file: {f}")
            
            # Sort by the integer name
            sorted_files.sort(key=lambda x: x[0])
            
            # Get just the sorted paths
            final_file_list = [path for _, path in sorted_files]

            if not final_file_list:
                print("No numerically named .jpg files found.")
                # You could show a message box here
                return

            # Pass the sorted list to the main window to switch pages
            self.main_window.show_viewer_page(final_file_list)

class ViewerWidget(QWidget):
    """Page 2: Image viewer with slider and action buttons."""
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.image_files = []
        self.current_pixmap = QPixmap()
        self.overlay_pixmap = QPixmap() # Pixmap for the mask overlay
        self.current_index = 0

        # SAM Model attributes
        self.device = None
        self.sam_mask_generator = None
        self.load_sam_model() # Load the model on initialization
        
        # Main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Top layout for "Back" button
        top_layout = QHBoxLayout()
        self.back_button = QPushButton("< Back to Upload")
        self.back_button.clicked.connect(self.main_window.show_upload_page)
        top_layout.addWidget(self.back_button)
        top_layout.addStretch(1)
        layout.addLayout(top_layout)

        # Image display label
        self.image_label = QLabel("Please load a folder first.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        layout.addWidget(self.image_label, 1) # Add stretch factor

        # Frame counter label
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.frame_label)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.slider_value_changed)
        layout.addWidget(self.slider)

        # Button layout
        button_layout = QHBoxLayout()
        
        self.generate_masks_button = QPushButton("Generate Masks")
        self.generate_masks_button.clicked.connect(
            lambda: self.main_window.show_results_page("Generate Masks")
        )
        
        self.identify_worms_button = QPushButton("Identify Worms")
        self.identify_worms_button.clicked.connect(
            self.on_identify_worms_clicked # Changed this connection
        )
        # Disable button if libraries failed to load
        if not _SAM_LOADED:
            self.identify_worms_button.setEnabled(False)
            self.identify_worms_button.setText("Identify Worms (Disabled)")
            self.identify_worms_button.setStyleSheet(
                "padding: 10px; font-size: 14px; background-color: #888; color: #ccc; border-radius: 5px;"
            )
        
        # Style buttons
        self.generate_masks_button.setStyleSheet(
            "padding: 10px; font-size: 14px; background-color: #28a745; color: white; border-radius: 5px;"
        )
        self.identify_worms_button.setStyleSheet(
            "padding: 10px; font-size: 14px; background-color: #dc3545; color: white; border-radius: 5px;"
        )

        button_layout.addStretch(1)
        button_layout.addWidget(self.generate_masks_button)
        button_layout.addWidget(self.identify_worms_button)
        button_layout.addStretch(1)
        
        layout.addLayout(button_layout)

    def set_image_files(self, files):
        """Receives the list of image files from the main window."""
        self.image_files = files
        self.overlay_pixmap = QPixmap() # Clear any old overlay
        if self.image_files:
            self.slider.setMaximum(len(self.image_files) - 1)
            self.slider.setMinimum(0)
            self.slider.setValue(0)
            self.load_image(0)
        else:
            self.slider.setMaximum(0)
            self.image_label.setText("No images found.")
            self.frame_label.setText("Frame: 0 / 0")

    def slider_value_changed(self, value):
        """Loads the image corresponding to the slider's value."""
        self.load_image(value)

    def load_image(self, index):
        """Loads, scales, and displays the image at the given index."""
        if 0 <= index < len(self.image_files):
            self.current_index = index
            self.current_pixmap = QPixmap(self.image_files[index])
            self.overlay_pixmap = QPixmap() # Clear any overlay when slider moves
            
            if self.current_pixmap.isNull():
                self.image_label.setText(f"Error loading image:\n{self.image_files[index]}")
            else:
                self.update_pixmap_scaling()
            
            # Update frame counter
            self.frame_label.setText(f"Frame: {index + 1} / {len(self.image_files)}")

    def update_pixmap_scaling(self):
        """Rescales the current pixmap (or overlay) to fit the label."""
        
        # Decide which pixmap to display: overlay if it exists, otherwise original
        pix_to_scale = self.current_pixmap
        if not self.overlay_pixmap.isNull():
            pix_to_scale = self.overlay_pixmap
            
        if not pix_to_scale.isNull():
            scaled_pixmap = pix_to_scale.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """
        Handles the widget being resized. We re-scale the image
        to fit the new label size.
        """
        super().resizeEvent(event)
        self.update_pixmap_scaling()

    def on_identify_worms_clicked(self):
        """
        Runs the mask generator and displays the result as a 50% overlay.
        """
        if not self.sam_mask_generator:
            print("SAM model not loaded. Cannot identify worms.")
            # Optionally show a user-facing error message
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Model Not Loaded")
            msg.setInformativeText("The SAM2 model failed to load. Please check console for errors.")
            msg.setWindowTitle("Error")
            msg.exec_()
            return

        if not self.image_files:
            print("No image loaded.")
            return

        current_image_path = self.image_files[self.current_index]
        original_pixmap = self.current_pixmap

        if original_pixmap.isNull():
            print("Original pixmap is null, cannot process.")
            return

        # 1. Run the REAL mask generator
        # This returns a QPixmap with transparent BG and colored masks
        print("Running SAM2... This may take a moment.")
        try:
            mask_pixmap = self.run_sam2_mask_generator(current_image_path)
            if mask_pixmap.isNull():
                print("Mask generation returned no masks.")
                return
        except Exception as e:
            print(f"Error during mask generation: {e}")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Processing Error")
            msg.setInformativeText(f"An error occurred while running the model: {e}")
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        
        print("Mask generation complete.")

        # 2. Create a new composite pixmap
        composite_pixmap = QPixmap(original_pixmap.size())
        composite_pixmap.fill(Qt.transparent) # Start with a transparent canvas

        # 3. Paint the overlay
        painter = QPainter(composite_pixmap)
        
        # Draw original image first (full opacity)
        painter.setOpacity(1.0)
        painter.drawPixmap(0, 0, original_pixmap)
        
        # Draw mask on top (50% opacity)
        painter.setOpacity(0.5)
        painter.drawPixmap(0, 0, mask_pixmap)
        
        painter.end()

        # 4. Store this composite pixmap and update the display
        self.overlay_pixmap = composite_pixmap
        self.update_pixmap_scaling()


    def load_sam_model(self):
        """
        Loads the SAM2 model and generator into memory.
        """
        if not _SAM_LOADED:
            print("Skipping model load due to import errors.")
            return
        
        try:
            print("Loading SAM2 model...")
            sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

            sam2 = build_sam2(model_cfg, sam2_checkpoint, device=self.device, apply_postprocessing=False)
            
            self.sam_mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2,
                points_per_side=64,
                points_per_batch=128,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.92,
                stability_score_offset=0.7,
                crop_n_layers=1,
                box_nms_thresh=0.7,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=25.0,
                use_m2m=True,
            )
            print("SAM2 model loaded successfully.")

        except Exception as e:
            print(f"--- FAILED TO LOAD SAM2 MODEL ---")
            print(f"Error: {e}")
            print("Please ensure sam2, torch, and all dependencies are installed.")
            print(f"And that checkpoint file exists: {sam2_checkpoint}")
            print(f"And that config file exists: {model_cfg}")
            self.sam_mask_generator = None


    def run_sam2_mask_generator(self, image_path):
        """
        *** REAL FUNCTION ***
        This runs the SAM2 model on the image at 'image_path'.
        
        It returns a QPixmap with masks drawn on a transparent background.
        """
        if not self.sam_mask_generator:
            print("SAM generator not available.")
            return QPixmap()

        print(f"--- Running SAM2 on: {image_path} ---")
        
        # 1. Read the image
        # cv2 reads as BGR
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Failed to read image: {image_path}")
            return QPixmap()
        
        # Convert to RGB for SAM
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Generate masks
        masks = self.sam_mask_generator.generate(image_rgb)
        
        if not masks:
            print("No masks found by SAM.")
            return QPixmap()
            
        # 3. Combine masks
        # 'segmentation' is a 2D boolean array (True where mask is)
        combined_masks = sum([mask['segmentation'] for mask in masks])
        
        # 4. Convert numpy array to QPixmap
        h, w = combined_masks.shape
        
        # Create an RGBA numpy array
        # Start with all transparent (Alpha = 0)
        mask_image_np = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Where the mask is True, set the color to pink (R=255, G=0, B=255, A=255)
        mask_image_np[combined_masks > 0] = [255, 0, 255, 255]

        # 5. Convert numpy array to QImage, then QPixmap
        bytes_per_line = 4 * w
        q_image = QImage(mask_image_np.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
        
        mask_pixmap = QPixmap.fromImage(q_image)
        
        print("--- SAM run complete. Returning mask pixmap. ---")
        
        return mask_pixmap


class ResultsWidget(QWidget):
    """Page 3: Placeholder for processing results."""
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

        self.title_label = QLabel("Results Page")
        title_font = self.title_label.font()
        title_font.setPointSize(24)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        self.info_label = QLabel(
            "Processing would happen here, and results would be displayed."
        )
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)
        
        layout.addStretch(1)

        self.back_button = QPushButton("Back to Viewer")
        self.back_button.setMinimumSize(150, 40)
        self.back_button.clicked.connect(
            lambda: self.main_window.stacked_widget.setCurrentIndex(1)
        )
        layout.addWidget(self.back_button, 0, Qt.AlignCenter)
        
        layout.addStretch(1)

    def update_title(self, from_button):
        """Updates the title based on which button was clicked."""
        self.title_label.setText(f"'{from_button}' Results")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())