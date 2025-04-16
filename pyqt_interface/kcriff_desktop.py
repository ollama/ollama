#!/usr/bin/env python3
"""
KC-Riff Desktop Application
A PyQt-based desktop interface for KC-Riff, the enhanced Ollama fork.
"""

import sys
import os
import signal
import json
import requests
import threading
import time
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QGridLayout, QProgressBar,
    QFrame, QScrollArea, QSizePolicy, QSpacerItem, QMessageBox
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QSize, QTimer
)
from PyQt5.QtGui import (
    QIcon, QPixmap, QFont, QColor, QPalette
)

# Constants
API_ENDPOINT = "http://localhost:5000"
MODELS_API = f"{API_ENDPOINT}/api/models"
DOWNLOAD_API = f"{API_ENDPOINT}/api/download"
REMOVE_API = f"{API_ENDPOINT}/api/remove"
STATUS_API = f"{API_ENDPOINT}/api/status"

class ModelDownloadThread(QThread):
    """Thread to download models without blocking the UI"""
    progress_updated = pyqtSignal(str, float)
    download_complete = pyqtSignal(str, bool)
    
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self._stop_requested = False
    
    def run(self):
        """Download a model and emit progress updates"""
        try:
            # Start the download
            response = requests.post(f"{DOWNLOAD_API}/{self.model_name}")
            if response.status_code != 200:
                self.download_complete.emit(self.model_name, False)
                return
                
            # Poll for status until complete
            completed = False
            while not completed and not self._stop_requested:
                try:
                    status_response = requests.get(f"{STATUS_API}/{self.model_name}")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        if "status" in status_data:
                            if status_data["status"] == "downloading":
                                progress = status_data.get("progress", 0)
                                self.progress_updated.emit(self.model_name, progress)
                            elif status_data["status"] == "completed":
                                completed = True
                                self.progress_updated.emit(self.model_name, 100)
                                self.download_complete.emit(self.model_name, True)
                            elif status_data["status"] == "failed":
                                self.download_complete.emit(self.model_name, False)
                                return
                except Exception as e:
                    print(f"Error polling status: {e}")
                
                # Sleep before next poll
                time.sleep(0.5)
                
            if self._stop_requested:
                # Attempt to cancel the download
                requests.post(f"{API_ENDPOINT}/api/cancel/{self.model_name}")
                self.download_complete.emit(self.model_name, False)
            
        except Exception as e:
            print(f"Download error: {e}")
            self.download_complete.emit(self.model_name, False)
    
    def stop(self):
        """Request the thread to stop"""
        self._stop_requested = True

class BatchDownloadThread(QThread):
    """Thread to download multiple models"""
    progress_updated = pyqtSignal(dict)  # {model_name: progress}
    batch_complete = pyqtSignal(bool, str)
    
    def __init__(self, model_names=None, download_recommended=False):
        super().__init__()
        self.model_names = model_names or []
        self.download_recommended = download_recommended
        self._stop_requested = False
        self.progress_dict = {}
    
    def run(self):
        """Download all models in the batch"""
        try:
            # If we're downloading recommended models, fetch the list first
            if self.download_recommended:
                try:
                    response = requests.get(MODELS_API)
                    if response.status_code == 200:
                        models_data = response.json()
                        self.model_names = [
                            model["name"] for model in models_data.get("models", [])
                            if model.get("kc_recommended", False)
                        ]
                    else:
                        self.batch_complete.emit(False, "Failed to get recommended models list")
                        return
                except Exception as e:
                    self.batch_complete.emit(False, f"Error fetching models: {e}")
                    return
            
            if not self.model_names:
                self.batch_complete.emit(True, "No models to download")
                return
            
            # Initialize progress dict
            for model in self.model_names:
                self.progress_dict[model] = 0
            
            # Start downloads sequentially
            for model_name in self.model_names:
                if self._stop_requested:
                    break
                
                # Start the download
                response = requests.post(f"{DOWNLOAD_API}/{model_name}")
                if response.status_code != 200:
                    self.progress_dict[model_name] = -1  # Mark as failed
                    self.progress_updated.emit(self.progress_dict.copy())
                    continue
                
                # Poll for status until complete
                completed = False
                while not completed and not self._stop_requested:
                    try:
                        status_response = requests.get(f"{STATUS_API}/{model_name}")
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            if "status" in status_data:
                                if status_data["status"] == "downloading":
                                    progress = status_data.get("progress", 0)
                                    self.progress_dict[model_name] = progress
                                    self.progress_updated.emit(self.progress_dict.copy())
                                elif status_data["status"] == "completed":
                                    completed = True
                                    self.progress_dict[model_name] = 100
                                    self.progress_updated.emit(self.progress_dict.copy())
                                elif status_data["status"] == "failed":
                                    self.progress_dict[model_name] = -1  # Mark as failed
                                    self.progress_updated.emit(self.progress_dict.copy())
                                    break
                    except Exception as e:
                        print(f"Error polling status: {e}")
                    
                    # Sleep before next poll
                    time.sleep(0.5)
                
                if self._stop_requested:
                    # Attempt to cancel the current download
                    requests.post(f"{API_ENDPOINT}/api/cancel/{model_name}")
                    break
            
            if self._stop_requested:
                self.batch_complete.emit(False, "Batch download was cancelled")
            else:
                # Count failures
                failures = sum(1 for progress in self.progress_dict.values() if progress < 100)
                if failures == 0:
                    self.batch_complete.emit(True, "All models downloaded successfully")
                else:
                    self.batch_complete.emit(
                        False, 
                        f"{failures} of {len(self.model_names)} models failed to download"
                    )
            
        except Exception as e:
            print(f"Batch download error: {e}")
            self.batch_complete.emit(False, str(e))
    
    def stop(self):
        """Request the thread to stop"""
        self._stop_requested = True

class ModelCard(QFrame):
    """Custom widget for displaying a model with download/remove buttons"""
    
    def __init__(self, model_data, parent=None, dark_mode=True):
        super().__init__(parent)
        self.model_data = model_data
        self.download_thread = None
        self.dark_mode = dark_mode
        self.parent_widget = parent
        
        # Set up UI
        self.setup_ui()
        
        # Update state based on current model data
        self.update_state()
    
    def setup_ui(self):
        """Create the UI components"""
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setMaximumWidth(300)
        
        # Apply styling based on whether this is a recommended model
        if self.model_data.get("kc_recommended", False):
            if self.dark_mode:
                self.setStyleSheet("""
                    QFrame {
                        background-color: #2C3E50;
                        border: 1px solid #3498DB;
                        border-radius: 5px;
                    }
                """)
            else:
                self.setStyleSheet("""
                    QFrame {
                        background-color: #EBF5FB;
                        border: 1px solid #3498DB;
                        border-radius: 5px;
                    }
                """)
        else:
            if self.dark_mode:
                self.setStyleSheet("""
                    QFrame {
                        background-color: #1E1E1E;
                        border: 1px solid #555555;
                        border-radius: 5px;
                    }
                """)
            else:
                self.setStyleSheet("""
                    QFrame {
                        background-color: #F8F9FA;
                        border: 1px solid #DDDDDD;
                        border-radius: 5px;
                    }
                """)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Model name
        self.name_label = QLabel(self.model_data.get("name", "Unknown Model"))
        name_font = QFont()
        name_font.setBold(True)
        name_font.setPointSize(12)
        self.name_label.setFont(name_font)
        
        # Description
        self.desc_label = QLabel(self.model_data.get("description", "No description available"))
        self.desc_label.setWordWrap(True)
        self.desc_label.setMaximumHeight(60)
        
        # Size and parameters info
        size_mb = self.model_data.get("size", 0) / 1000000
        params_b = self.model_data.get("parameters", 0) / 1000000000
        self.info_label = QLabel(f"Size: {size_mb:.1f} MB | Parameters: {params_b:.1f}B")
        
        # Tags
        tags = self.model_data.get("tags", [])
        self.tags_label = QLabel(" ".join([f"#{tag}" for tag in tags]))
        self.tags_label.setWordWrap(True)
        
        # Progress bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
        # Action button (download or remove)
        self.action_button = QPushButton("Download")
        self.action_button.clicked.connect(self.on_action_button_clicked)
        
        # Add all widgets to layout
        layout.addWidget(self.name_label)
        layout.addWidget(self.desc_label)
        layout.addWidget(self.info_label)
        layout.addWidget(self.tags_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.action_button)
    
    def update_state(self):
        """Update the UI based on model state"""
        is_downloaded = self.model_data.get("is_downloaded", False)
        
        if is_downloaded:
            self.action_button.setText("Remove")
            self.action_button.setStyleSheet("background-color: #E74C3C;")
            self.progress_bar.setVisible(False)
        else:
            self.action_button.setText("Download")
            self.action_button.setStyleSheet("")
            # Progress bar visibility depends on whether a download is in progress
    
    def update_model_data(self, model_data):
        """Update with new model data"""
        self.model_data = model_data
        
        # Update UI elements
        self.name_label.setText(model_data.get("name", "Unknown Model"))
        self.desc_label.setText(model_data.get("description", "No description available"))
        
        size_mb = model_data.get("size", 0) / 1000000
        params_b = model_data.get("parameters", 0) / 1000000000
        self.info_label.setText(f"Size: {size_mb:.1f} MB | Parameters: {params_b:.1f}B")
        
        tags = model_data.get("tags", [])
        self.tags_label.setText(" ".join([f"#{tag}" for tag in tags]))
        
        # Update state (downloaded status, button text, etc.)
        self.update_state()
    
    def on_action_button_clicked(self):
        """Handle download or remove button click"""
        model_name = self.model_data.get("name", "")
        if not model_name:
            return
        
        if self.model_data.get("is_downloaded", False):
            self.remove_model()
        else:
            self.download_model()
    
    def download_model(self):
        """Start downloading the model"""
        if self.download_thread and self.download_thread.isRunning():
            return  # Already downloading
        
        model_name = self.model_data.get("name", "")
        
        # Update UI
        self.action_button.setEnabled(False)
        self.action_button.setText("Starting...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Start download thread
        self.download_thread = ModelDownloadThread(model_name)
        self.download_thread.progress_updated.connect(self.update_progress)
        self.download_thread.download_complete.connect(self.download_finished)
        self.download_thread.start()
    
    def update_progress(self, model_name, progress):
        """Update the progress bar value"""
        if model_name == self.model_data.get("name", ""):
            self.progress_bar.setValue(int(progress))
            self.action_button.setText(f"Downloading... {int(progress)}%")
    
    def download_finished(self, model_name, success):
        """Handle download completion"""
        if model_name != self.model_data.get("name", ""):
            return
        
        self.progress_bar.setVisible(False)
        self.action_button.setEnabled(True)
        
        if success:
            # Update model data and UI
            self.model_data["is_downloaded"] = True
            self.update_state()
            
            # Inform parent to refresh all models (as download status has changed)
            if hasattr(self.parent_widget, 'refresh_models'):
                self.parent_widget.refresh_models()
        else:
            self.action_button.setText("Download Failed - Retry")
            QMessageBox.warning(
                self,
                "Download Failed",
                f"Failed to download model {model_name}. Please try again."
            )
    
    def remove_model(self):
        """Remove the downloaded model"""
        model_name = self.model_data.get("name", "")
        if not model_name:
            return
        
        # Confirm with the user
        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove the {model_name} model?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Update UI
        self.action_button.setEnabled(False)
        self.action_button.setText("Removing...")
        
        try:
            # Send remove request
            response = requests.post(f"{REMOVE_API}/{model_name}")
            
            if response.status_code == 200:
                # Update model data and UI
                self.model_data["is_downloaded"] = False
                self.update_state()
                
                # Inform parent to refresh all models
                if hasattr(self.parent_widget, 'refresh_models'):
                    self.parent_widget.refresh_models()
            else:
                # Handle failure
                QMessageBox.warning(
                    self,
                    "Removal Failed",
                    f"Failed to remove model {model_name}. Status code: {response.status_code}"
                )
        except Exception as e:
            QMessageBox.warning(
                self,
                "Removal Failed",
                f"Error removing model {model_name}: {str(e)}"
            )
        
        self.action_button.setEnabled(True)
        self.action_button.setText("Download")

class KCRiffDesktop(QMainWindow):
    """Main application window for KC-Riff Desktop"""
    
    def __init__(self):
        super().__init__()
        
        # Application state
        self.models = []
        self.dark_mode = True
        self.batch_download_thread = None
        
        # Set up UI
        self.setup_ui()
        
        # Load models
        self.load_models()
        
        # Set up auto-refresh timer (every 30 seconds)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_models)
        self.refresh_timer.start(30000)  # 30 seconds
    
    def setup_ui(self):
        """Set up the main UI"""
        self.setWindowTitle("KC-Riff Model Manager")
        self.setMinimumSize(800, 600)
        
        # Main widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Header with logo and theme toggle
        header_layout = QHBoxLayout()
        
        # Logo/title - You might want to add an actual logo image here
        logo_label = QLabel("KC-Riff")
        logo_font = QFont()
        logo_font.setBold(True)
        logo_font.setPointSize(18)
        logo_label.setFont(logo_font)
        header_layout.addWidget(logo_label)
        
        header_layout.addStretch()
        
        # Theme toggle button
        self.theme_button = QPushButton("Toggle Light/Dark Mode")
        self.theme_button.clicked.connect(self.toggle_theme)
        header_layout.addWidget(self.theme_button)
        
        main_layout.addLayout(header_layout)
        
        # Tab widget for different model categories
        self.tabs = QTabWidget()
        
        # Create tabs
        self.recommended_tab = QWidget()
        self.standard_tab = QWidget()
        self.advanced_tab = QWidget()
        
        # Set up layouts for each tab
        recommended_layout = QVBoxLayout(self.recommended_tab)
        standard_layout = QVBoxLayout(self.standard_tab)
        advanced_layout = QVBoxLayout(self.advanced_tab)
        
        # Add a "Download all recommended" button to the recommended tab
        download_all_btn = QPushButton("Download All Recommended Models")
        download_all_btn.clicked.connect(self.download_all_recommended)
        recommended_layout.addWidget(download_all_btn)
        
        # Create scroll areas and grid layouts for models
        # Recommended models
        self.recommended_scroll = QScrollArea()
        self.recommended_scroll.setWidgetResizable(True)
        self.recommended_container = QWidget()
        self.recommended_grid = QGridLayout(self.recommended_container)
        self.recommended_scroll.setWidget(self.recommended_container)
        recommended_layout.addWidget(self.recommended_scroll)
        
        # Standard models
        self.standard_scroll = QScrollArea()
        self.standard_scroll.setWidgetResizable(True)
        self.standard_container = QWidget()
        self.standard_grid = QGridLayout(self.standard_container)
        self.standard_scroll.setWidget(self.standard_container)
        standard_layout.addWidget(self.standard_scroll)
        
        # Advanced models
        self.advanced_scroll = QScrollArea()
        self.advanced_scroll.setWidgetResizable(True)
        self.advanced_container = QWidget()
        self.advanced_grid = QGridLayout(self.advanced_container)
        self.advanced_scroll.setWidget(self.advanced_container)
        advanced_layout.addWidget(self.advanced_scroll)
        
        # Add tabs to tab widget
        self.tabs.addTab(self.recommended_tab, "Recommended Models")
        self.tabs.addTab(self.standard_tab, "Standard Models")
        self.tabs.addTab(self.advanced_tab, "Advanced Models")
        
        main_layout.addWidget(self.tabs)
        
        # Status bar with refresh button
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_models)
        status_layout.addWidget(refresh_button)
        
        main_layout.addLayout(status_layout)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Apply theme
        self.apply_theme()
    
    def apply_theme(self):
        """Apply the current theme to the application"""
        if self.dark_mode:
            # Dark theme
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #121212;
                    color: #FFFFFF;
                }
                QTabWidget::pane {
                    border: 1px solid #555555;
                    background-color: #1E1E1E;
                }
                QTabBar::tab {
                    background-color: #2D2D2D;
                    color: #BBBBBB;
                    border: 1px solid #555555;
                    padding: 5px 10px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
                QTabBar::tab:selected {
                    background-color: #3498DB;
                    color: #FFFFFF;
                }
                QPushButton {
                    background-color: #2C3E50;
                    color: #FFFFFF;
                    border: 1px solid #3498DB;
                    padding: 5px 10px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #34495E;
                }
                QPushButton:pressed {
                    background-color: #1ABC9C;
                }
                QScrollArea {
                    background-color: #1E1E1E;
                    border: 1px solid #555555;
                }
                QLabel {
                    color: #FFFFFF;
                }
                QProgressBar {
                    background-color: #1E1E1E;
                    color: #FFFFFF;
                    border: 1px solid #555555;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #3498DB;
                }
            """)
        else:
            # Light theme
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #FFFFFF;
                    color: #333333;
                }
                QTabWidget::pane {
                    border: 1px solid #DDDDDD;
                    background-color: #F8F9FA;
                }
                QTabBar::tab {
                    background-color: #F0F0F0;
                    color: #555555;
                    border: 1px solid #DDDDDD;
                    padding: 5px 10px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
                QTabBar::tab:selected {
                    background-color: #3498DB;
                    color: #FFFFFF;
                }
                QPushButton {
                    background-color: #ECF0F1;
                    color: #333333;
                    border: 1px solid #BDC3C7;
                    padding: 5px 10px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #D6DBDF;
                }
                QPushButton:pressed {
                    background-color: #1ABC9C;
                    color: #FFFFFF;
                }
                QScrollArea {
                    background-color: #F8F9FA;
                    border: 1px solid #DDDDDD;
                }
                QLabel {
                    color: #333333;
                }
                QProgressBar {
                    background-color: #F8F9FA;
                    color: #333333;
                    border: 1px solid #DDDDDD;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #3498DB;
                }
            """)
    
    def toggle_theme(self):
        """Switch between light and dark themes"""
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        
        # Also update all model cards
        for i in range(self.recommended_grid.count()):
            widget = self.recommended_grid.itemAt(i).widget()
            if isinstance(widget, ModelCard):
                widget.dark_mode = self.dark_mode
                widget.setup_ui()
        
        for i in range(self.standard_grid.count()):
            widget = self.standard_grid.itemAt(i).widget()
            if isinstance(widget, ModelCard):
                widget.dark_mode = self.dark_mode
                widget.setup_ui()
        
        for i in range(self.advanced_grid.count()):
            widget = self.advanced_grid.itemAt(i).widget()
            if isinstance(widget, ModelCard):
                widget.dark_mode = self.dark_mode
                widget.setup_ui()
    
    def load_models(self):
        """Load models from the API"""
        self.status_label.setText("Loading models...")
        
        try:
            response = requests.get(MODELS_API)
            if response.status_code == 200:
                data = response.json()
                self.models = data.get("models", [])
                
                # Sort models by recommended status then by name
                self.models.sort(key=lambda x: (not x.get("kc_recommended", False), x.get("name", "")))
                
                # Display models in their respective tabs
                self.clear_model_grids()
                
                # Filter models by category
                recommended_models = [m for m in self.models if m.get("kc_recommended", False)]
                standard_models = [m for m in self.models if m.get("category", "") == "standard" and not m.get("kc_recommended", False)]
                advanced_models = [m for m in self.models if m.get("category", "") == "advanced" and not m.get("kc_recommended", False)]
                
                # Display models
                self.display_models(recommended_models, self.recommended_grid, is_recommended=True)
                self.display_models(standard_models, self.standard_grid)
                self.display_models(advanced_models, self.advanced_grid)
                
                self.status_label.setText(f"Loaded {len(self.models)} models")
            else:
                self.status_label.setText(f"Error loading models: {response.status_code}")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
    
    def clear_model_grids(self):
        """Clear the model grid layouts"""
        # Remove all widgets from grid layouts
        for i in reversed(range(self.recommended_grid.count())):
            widget = self.recommended_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        for i in reversed(range(self.standard_grid.count())):
            widget = self.standard_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        for i in reversed(range(self.advanced_grid.count())):
            widget = self.advanced_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
    
    def display_models(self, models, grid_layout, is_recommended=False):
        """Display models in the specified grid layout"""
        # Add model cards to grid
        row, col = 0, 0
        max_cols = 3  # Maximum columns in the grid
        
        for model in models:
            model_card = ModelCard(model, self, self.dark_mode)
            grid_layout.addWidget(model_card, row, col)
            
            # Move to next column or row
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
    
    def download_all_recommended(self):
        """Download all recommended models at once"""
        if self.batch_download_thread and self.batch_download_thread.isRunning():
            QMessageBox.information(
                self,
                "Download in Progress",
                "A batch download is already in progress. Please wait for it to complete."
            )
            return
        
        reply = QMessageBox.question(
            self,
            "Download All Recommended Models",
            "This will download all KC-Riff recommended models. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        self.status_label.setText("Starting batch download of recommended models...")
        
        # Start batch download thread
        self.batch_download_thread = BatchDownloadThread(download_recommended=True)
        self.batch_download_thread.progress_updated.connect(self.update_batch_progress)
        self.batch_download_thread.batch_complete.connect(self.batch_download_finished)
        self.batch_download_thread.start()
    
    def update_batch_progress(self, progress_dict):
        """Update progress for batch download"""
        model_count = len(progress_dict)
        completed = sum(1 for progress in progress_dict.values() if progress == 100)
        failed = sum(1 for progress in progress_dict.values() if progress == -1)
        in_progress = model_count - completed - failed
        
        self.status_label.setText(
            f"Batch download: {completed} completed, {in_progress} in progress, {failed} failed"
        )
        
        # Also update the individual model cards
        for i in range(self.recommended_grid.count()):
            widget = self.recommended_grid.itemAt(i).widget()
            if isinstance(widget, ModelCard):
                model_name = widget.model_data.get("name", "")
                if model_name in progress_dict:
                    if progress_dict[model_name] == 100:
                        # Model is downloaded
                        widget.model_data["is_downloaded"] = True
                        widget.update_state()
                    elif progress_dict[model_name] > 0:
                        # Model is being downloaded
                        widget.progress_bar.setValue(int(progress_dict[model_name]))
                        widget.progress_bar.setVisible(True)
                        widget.action_button.setText(f"Downloading... {int(progress_dict[model_name])}%")
                        widget.action_button.setEnabled(False)
    
    def batch_download_finished(self, success, message):
        """Handle batch download completion"""
        if success:
            self.status_label.setText(f"Batch download complete: {message}")
            QMessageBox.information(
                self,
                "Batch Download Complete",
                message
            )
        else:
            self.status_label.setText(f"Batch download issue: {message}")
            QMessageBox.warning(
                self,
                "Batch Download Issue",
                message
            )
        
        # Refresh models to get updated status
        self.refresh_models()
    
    def refresh_models(self):
        """Refresh model data"""
        self.load_models()
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop any active downloads
        if self.batch_download_thread and self.batch_download_thread.isRunning():
            self.batch_download_thread.stop()
        
        # Check each model card for active downloads
        for grid in [self.recommended_grid, self.standard_grid, self.advanced_grid]:
            for i in range(grid.count()):
                widget = grid.itemAt(i).widget()
                if isinstance(widget, ModelCard) and widget.download_thread and widget.download_thread.isRunning():
                    widget.download_thread.stop()
        
        event.accept()

def main():
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = KCRiffDesktop()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()