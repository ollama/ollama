#!/usr/bin/env python3
"""
KC-Riff Desktop Application (PyQt6 Version)
A PyQt6-based desktop interface for KC-Riff, the enhanced Ollama fork.
"""

import sys
import os
import signal
import json
import requests
import threading
import time
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QGridLayout, QProgressBar,
    QFrame, QScrollArea, QSizePolicy, QSpacerItem, QMessageBox
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QSize, QTimer
)
from PyQt6.QtGui import (
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
        # In PyQt6, QFrame.Shape and QFrame.Shadow are enums
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
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
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Update UI
        self.action_button.setEnabled(False)
        self.action_button.setText("Removing...")
        
        # Call the API
        try:
            response = requests.post(f"{REMOVE_API}/{model_name}")
            if response.status_code == 200:
                # Update model data and UI
                self.model_data["is_downloaded"] = False
                self.update_state()
                
                # Inform parent to refresh all models
                if hasattr(self.parent_widget, 'refresh_models'):
                    self.parent_widget.refresh_models()
            else:
                self.action_button.setEnabled(True)
                self.action_button.setText("Remove")
                QMessageBox.warning(
                    self,
                    "Removal Failed",
                    f"Failed to remove model {model_name}. Please try again."
                )
        except Exception as e:
            print(f"Error removing model: {e}")
            self.action_button.setEnabled(True)
            self.action_button.setText("Remove")
            QMessageBox.warning(
                self,
                "Removal Failed",
                f"Failed to remove model {model_name}: {str(e)}"
            )

class KCRiffDesktop(QMainWindow):
    """Main application window for KC-Riff Desktop"""
    
    def __init__(self):
        super().__init__()
        self.dark_mode = True
        self.batch_download_thread = None
        self.model_cards = {}  # To track all model cards for updating
        
        # Set up the UI
        self.setup_ui()
        
        # Start model loading
        self.load_models()
        
        # Set up refresh timer (every 30 seconds)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_models)
        self.refresh_timer.start(30000)  # 30 seconds
    
    def setup_ui(self):
        """Set up the main UI"""
        self.setWindowTitle("KC-Riff Desktop")
        self.setGeometry(100, 100, 1000, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Create header with logo and title
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        
        # Logo (placeholder for now)
        logo_label = QLabel()
        
        # Load the appropriate logo based on theme
        logo_path = "theme/assets/kc-riff-midnight-logo.svg" if self.dark_mode else "theme/assets/kc-riff-classic-logo.svg"
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio))
        else:
            # Fallback text if logo not found
            logo_label.setText("KC-Riff")
            logo_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        
        # Title and description
        title_widget = QWidget()
        title_layout = QVBoxLayout(title_widget)
        
        title_label = QLabel("KC-Riff")
        title_label.setStyleSheet("font-size: 28px; font-weight: bold;")
        
        desc_label = QLabel("KillChaos AI Model Management")
        desc_label.setStyleSheet("font-size: 16px;")
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(desc_label)
        
        # Control buttons
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        
        refresh_button = QPushButton("Refresh Models")
        refresh_button.clicked.connect(self.refresh_models)
        
        theme_button = QPushButton("Toggle Theme")
        theme_button.clicked.connect(self.toggle_theme)
        
        batch_download_button = QPushButton("Download All Recommended")
        batch_download_button.clicked.connect(self.download_all_recommended)
        
        control_layout.addWidget(refresh_button)
        control_layout.addWidget(theme_button)
        control_layout.addWidget(batch_download_button)
        
        # Add all header components
        header_layout.addWidget(logo_label)
        header_layout.addWidget(title_widget)
        header_layout.addStretch()
        header_layout.addWidget(control_widget)
        
        # Create tab widget for different model categories
        self.tabs = QTabWidget()
        
        # Create scrollable areas for each tab
        self.recommended_scroll = QScrollArea()
        self.recommended_scroll.setWidgetResizable(True)
        self.recommended_content = QWidget()
        self.recommended_grid = QGridLayout(self.recommended_content)
        self.recommended_scroll.setWidget(self.recommended_content)
        
        self.standard_scroll = QScrollArea()
        self.standard_scroll.setWidgetResizable(True)
        self.standard_content = QWidget()
        self.standard_grid = QGridLayout(self.standard_content)
        self.standard_scroll.setWidget(self.standard_content)
        
        self.advanced_scroll = QScrollArea()
        self.advanced_scroll.setWidgetResizable(True)
        self.advanced_content = QWidget()
        self.advanced_grid = QGridLayout(self.advanced_content)
        self.advanced_scroll.setWidget(self.advanced_content)
        
        # Add tabs
        self.tabs.addTab(self.recommended_scroll, "Recommended")
        self.tabs.addTab(self.standard_scroll, "Standard")
        self.tabs.addTab(self.advanced_scroll, "Advanced")
        
        # Add header and tabs to main layout
        main_layout.addWidget(header_widget)
        main_layout.addWidget(self.tabs)
        
        # Set the main widget
        self.setCentralWidget(main_widget)
        
        # Apply the initial theme
        self.apply_theme()
    
    def apply_theme(self):
        """Apply the current theme to the application"""
        if self.dark_mode:
            # Dark theme
            self.setStyleSheet("""
                QMainWindow, QTabWidget, QScrollArea, QWidget {
                    background-color: #121212;
                    color: #EEEEEE;
                }
                QTabWidget::pane {
                    border: 1px solid #444444;
                }
                QTabBar::tab {
                    background-color: #2D2D2D;
                    color: #EEEEEE;
                    padding: 8px 20px;
                    border: 1px solid #444444;
                }
                QTabBar::tab:selected {
                    background-color: #3498DB;
                }
                QPushButton {
                    background-color: #3498DB;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #2980B9;
                }
                QPushButton:disabled {
                    background-color: #555555;
                }
                QProgressBar {
                    border: 1px solid #444444;
                    border-radius: 3px;
                    background-color: #2D2D2D;
                    text-align: center;
                    color: white;
                }
                QProgressBar::chunk {
                    background-color: #3498DB;
                    width: 1px;
                }
                QLabel {
                    color: #EEEEEE;
                }
            """)
        else:
            # Light theme
            self.setStyleSheet("""
                QMainWindow, QTabWidget, QScrollArea, QWidget {
                    background-color: #F8F9FA;
                    color: #212529;
                }
                QTabWidget::pane {
                    border: 1px solid #DEE2E6;
                }
                QTabBar::tab {
                    background-color: #E9ECEF;
                    color: #495057;
                    padding: 8px 20px;
                    border: 1px solid #DEE2E6;
                }
                QTabBar::tab:selected {
                    background-color: #3498DB;
                    color: white;
                }
                QPushButton {
                    background-color: #3498DB;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #2980B9;
                }
                QPushButton:disabled {
                    background-color: #CED4DA;
                }
                QProgressBar {
                    border: 1px solid #DEE2E6;
                    border-radius: 3px;
                    background-color: #E9ECEF;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #3498DB;
                    width: 1px;
                }
                QLabel {
                    color: #212529;
                }
            """)
    
    def toggle_theme(self):
        """Switch between light and dark themes"""
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        
        # Update all model cards
        for card in self.model_cards.values():
            card.dark_mode = self.dark_mode
            card.setup_ui()  # Rebuild UI with new theme
            card.update_state()
    
    def load_models(self):
        """Load models from the API"""
        try:
            response = requests.get(MODELS_API)
            if response.status_code == 200:
                models_data = response.json()
                all_models = models_data.get("models", [])
                
                # Store models by category
                recommended_models = [m for m in all_models if m.get("kc_recommended", False)]
                standard_models = [m for m in all_models if m.get("category", "") == "standard" and not m.get("kc_recommended", False)]
                advanced_models = [m for m in all_models if m.get("category", "") == "advanced" and not m.get("kc_recommended", False)]
                
                # Clear and repopulate grids
                self.clear_model_grids()
                
                # Display models in their respective tabs
                self.display_models(recommended_models, self.recommended_grid, is_recommended=True)
                self.display_models(standard_models, self.standard_grid)
                self.display_models(advanced_models, self.advanced_grid)
            else:
                print(f"Error fetching models: {response.status_code}")
                QMessageBox.warning(
                    self,
                    "Connection Error",
                    f"Failed to load models (Status {response.status_code}). Is the KC-Riff server running?"
                )
        except Exception as e:
            print(f"Error loading models: {e}")
            QMessageBox.warning(
                self,
                "Connection Error",
                f"Failed to connect to KC-Riff server: {str(e)}"
            )
    
    def clear_model_grids(self):
        """Clear the model grid layouts"""
        # Function to clear a grid layout
        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
        
        clear_layout(self.recommended_grid)
        clear_layout(self.standard_grid)
        clear_layout(self.advanced_grid)
        
        # Reset tracking
        self.model_cards = {}
    
    def display_models(self, models, grid_layout, is_recommended=False):
        """Display models in the specified grid layout"""
        if not models:
            # No models message
            label = QLabel("No models available in this category")
            label.setStyleSheet("font-size: 14px; color: #888888;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid_layout.addWidget(label, 0, 0, 1, 3)
            return
        
        # Add models to grid (3 columns)
        for i, model in enumerate(models):
            row = i // 3
            col = i % 3
            
            # Create model card
            card = ModelCard(model, self, self.dark_mode)
            
            # Store for future reference
            self.model_cards[model.get("name", f"unknown_{i}")] = card
            
            # Add to grid
            grid_layout.addWidget(card, row, col)
        
        # Add stretch at the end
        grid_layout.setRowStretch(grid_layout.rowCount(), 1)
    
    def download_all_recommended(self):
        """Download all recommended models at once"""
        if self.batch_download_thread and self.batch_download_thread.isRunning():
            # Already downloading
            QMessageBox.information(
                self,
                "Download in Progress",
                "A batch download is already in progress."
            )
            return
        
        # Confirm with the user
        reply = QMessageBox.question(
            self,
            "Confirm Batch Download",
            "Do you want to download all recommended models? This may take significant time and disk space.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Start batch download
        self.batch_download_thread = BatchDownloadThread(download_recommended=True)
        self.batch_download_thread.progress_updated.connect(self.update_batch_progress)
        self.batch_download_thread.batch_complete.connect(self.batch_download_finished)
        self.batch_download_thread.start()
        
        # Notify user
        QMessageBox.information(
            self,
            "Batch Download Started",
            "Downloading all recommended models. You can continue using the application."
        )
    
    def update_batch_progress(self, progress_dict):
        """Update progress for batch download"""
        # Update progress on corresponding model cards
        for model_name, progress in progress_dict.items():
            if model_name in self.model_cards:
                card = self.model_cards[model_name]
                if progress >= 0:  # Not failed
                    card.update_progress(model_name, progress)
                else:
                    # Handle failed download
                    card.download_finished(model_name, False)
    
    def batch_download_finished(self, success, message):
        """Handle batch download completion"""
        self.refresh_models()  # Refresh to show updated state
        
        if success:
            QMessageBox.information(
                self,
                "Batch Download Complete",
                message
            )
        else:
            QMessageBox.warning(
                self,
                "Batch Download Issue",
                message
            )
    
    def refresh_models(self):
        """Refresh model data"""
        self.load_models()
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop any downloads in progress
        if self.batch_download_thread and self.batch_download_thread.isRunning():
            self.batch_download_thread.stop()
        
        for card in self.model_cards.values():
            if card.download_thread and card.download_thread.isRunning():
                card.download_thread.stop()
        
        # Accept the close event
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = KCRiffDesktop()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()