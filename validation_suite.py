#!/usr/bin/env python3
"""
Hardware Validation Suite for Drone FPV Tracking Systems

Simulates a Test Engineer's workflow for stress-testing target tracking algorithms
using real-world analog video signal degradation from FPV drone goggles.

Author: Hardware Validation Engineer
Purpose: Camera Algorithm QA & Field Testing
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys
import os
from pathlib import Path

# YOLO26 import (optional - only loaded if needed)
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass


class YOLO26Tracker:
    """
    YOLO26-based detection tracker wrapper.
    
    Unlike traditional trackers, YOLO26 re-detects objects in every frame.
    This makes it robust to rotation and able to re-acquire lost objects.
    
    Trade-offs:
    - Pro: Handles rotation, can recover from occlusion
    - Pro: Knows object class (not just appearance)
    - Con: Slower than correlation-based trackers
    - Con: Requires more compute (GPU recommended)
    """
    
    def __init__(self, model_size='n', target_class=None):
        """
        Initialize YOLO26 detector.
        
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
            target_class: Optional class ID to track (None = any object)
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("YOLO not available. Install with: pip install ultralytics")
        
        print(f"   Loading YOLO26-{model_size} model (first run downloads ~6MB)...")
        self.model = YOLO(f'yolo26{model_size}.pt')
        self.target_class = target_class
        self.target_bbox = None
        self.confidence_threshold = 0.25
        
    def init(self, frame, bbox):
        """
        Initialize with a bounding box (matches OpenCV tracker interface).
        
        For YOLO, we use this to identify WHICH detected object to track
        by finding the detection that overlaps most with the initial bbox.
        """
        self.target_bbox = bbox
        x, y, w, h = bbox
        self.target_center = (x + w // 2, y + h // 2)
        
        # Run initial detection to identify target class
        results = self.model(frame, verbose=False)
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Find detection closest to initial bbox
            best_iou = 0
            for box in results[0].boxes:
                det_bbox = box.xyxy[0].cpu().numpy()
                iou = self._calculate_iou(bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    self.target_class = int(box.cls[0])
            if best_iou > 0:
                class_name = self.model.names.get(self.target_class, 'unknown')
                print(f"   YOLO26 identified target as: {class_name} (class {self.target_class})")
        
        return True
    
    def update(self, frame):
        """
        Detect objects and find the one closest to our target.
        
        Returns:
            (success, bbox): Matches OpenCV tracker interface
        """
        results = self.model(frame, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return False, None
        
        # Find the detection closest to our last known position
        best_match = None
        best_distance = float('inf')
        
        for box in results[0].boxes:
            # Filter by confidence
            if box.conf[0] < self.confidence_threshold:
                continue
            
            # If we identified a target class, prefer that class
            if self.target_class is not None and int(box.cls[0]) != self.target_class:
                continue
            
            # Get bbox in (x, y, w, h) format
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            det_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Distance from our target center
            distance = np.sqrt(
                (det_center[0] - self.target_center[0])**2 +
                (det_center[1] - self.target_center[1])**2
            )
            
            if distance < best_distance:
                best_distance = distance
                best_match = (x1, y1, x2 - x1, y2 - y1)  # Convert to (x, y, w, h)
                self.target_center = det_center
        
        if best_match is not None:
            self.target_bbox = best_match
            return True, best_match
        
        return False, None
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bboxes."""
        # bbox1 is (x, y, w, h), bbox2 is (x1, y1, x2, y2)
        x1_1, y1_1 = bbox1[0], bbox1[1]
        x2_1, y2_1 = bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        return inter_area / (box1_area + box2_area - inter_area)


class HybridTracker:
    """
    Hybrid tracker combining fast correlation tracking with YOLO26 verification.
    
    Strategy:
    - Use KCF for fast frame-to-frame tracking (~100 FPS)
    - Run YOLO26 every N frames to verify/correct position
    - If KCF loses track, immediately use YOLO26 to re-acquire
    
    This is how production systems (Tesla Autopilot, etc.) work!
    """
    
    def __init__(self, yolo_interval=15, error_threshold=200):
        """
        Initialize hybrid tracker.
        
        Args:
            yolo_interval: Run YOLO every N frames (default: 30 = ~1 second at 30fps)
            error_threshold: If error exceeds this, trigger YOLO check
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("YOLO not available for hybrid tracking")
        
        print(f"   Initializing Hybrid Tracker (KCF + YOLO26)")
        print(f"   YOLO check interval: every {yolo_interval} frames")
        
        self.yolo_interval = yolo_interval
        self.error_threshold = error_threshold
        
        # Initialize fast tracker (use MIL as it's available in all OpenCV versions)
        self.fast_tracker = cv2.TrackerMIL_create()
        self.yolo_detector = None  # Lazy load on first use
        
        self.frame_count = 0
        self.last_yolo_bbox = None
        self.target_class = None
        self.initial_center = None
        self.yolo_checks = 0
        self.yolo_corrections = 0
        
    def init(self, frame, bbox):
        """Initialize with bounding box."""
        self.initial_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        self.original_bbox = bbox  # Store original for re-identification
        self.original_size = (bbox[2], bbox[3])  # Store original size
        
        # Initialize fast tracker
        self.fast_tracker.init(frame, bbox)
        self.last_yolo_bbox = bbox
        
        # Lazy load YOLO (heavy, only load when needed)
        print(f"   Loading YOLO26-n model for verification...")
        self.yolo_detector = YOLO('yolo26n.pt')
        
        # Identify target class from initial selection
        results = self.yolo_detector(frame, verbose=False)
        if len(results) > 0 and len(results[0].boxes) > 0:
            best_iou = 0
            for box in results[0].boxes:
                det_bbox = box.xyxy[0].cpu().numpy()
                iou = self._calculate_iou(bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    self.target_class = int(box.cls[0])
            if self.target_class is not None:
                class_name = self.yolo_detector.names.get(self.target_class, 'unknown')
                print(f"   YOLO26 locked on class: {class_name}")
                print(f"   Will ONLY track this class, ignoring others")
        
        return True
    
    def update(self, frame):
        """
        Update tracking - fast tracker with periodic YOLO verification.
        """
        self.frame_count += 1
        
        # First, try fast KCF tracking
        success, bbox = self.fast_tracker.update(frame)
        
        # Decide if we need YOLO verification
        run_yolo = False
        reason = ""
        
        if not success:
            run_yolo = True
            reason = "KCF lost track"
        elif self.frame_count % self.yolo_interval == 0:
            run_yolo = True
            reason = f"Periodic check (frame {self.frame_count})"
        elif success:
            # Check if we've drifted too far from last YOLO position
            x, y, w, h = [int(v) for v in bbox]
            current_center = (x + w // 2, y + h // 2)
            last_center = (self.last_yolo_bbox[0] + self.last_yolo_bbox[2] // 2,
                          self.last_yolo_bbox[1] + self.last_yolo_bbox[3] // 2)
            drift = np.sqrt((current_center[0] - last_center[0])**2 + 
                           (current_center[1] - last_center[1])**2)
            if drift > self.error_threshold:
                run_yolo = True
                reason = f"Drift exceeded {self.error_threshold}px"
        
        # Run YOLO if needed
        if run_yolo and self.yolo_detector is not None:
            self.yolo_checks += 1
            yolo_success, yolo_bbox = self._yolo_detect(frame)
            
            if yolo_success:
                # YOLO found the target - update both trackers
                self.last_yolo_bbox = yolo_bbox
                
                # Re-initialize MIL with YOLO's more accurate position
                self.fast_tracker = cv2.TrackerMIL_create()
                self.fast_tracker.init(frame, tuple(int(v) for v in yolo_bbox))
                
                if not success or reason != f"Periodic check (frame {self.frame_count})":
                    self.yolo_corrections += 1
                
                return True, yolo_bbox
            elif not success:
                # Both failed
                return False, None
        
        return success, bbox
    
    def _yolo_detect(self, frame):
        """Run YOLO detection and find the SAME target we locked onto."""
        results = self.yolo_detector(frame, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return False, None
        
        # Find detection that best matches our original target
        best_match = None
        best_score = -1
        last_center = (self.last_yolo_bbox[0] + self.last_yolo_bbox[2] // 2,
                      self.last_yolo_bbox[1] + self.last_yolo_bbox[3] // 2)
        
        for box in results[0].boxes:
            # Must have high confidence
            if box.conf[0] < 0.4:
                continue
            # MUST be same class as original target
            if self.target_class is not None and int(box.cls[0]) != self.target_class:
                continue
            
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            det_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            det_size = (x2 - x1, y2 - y1)
            
            # Score based on: distance + size similarity
            distance = np.sqrt((det_center[0] - last_center[0])**2 +
                              (det_center[1] - last_center[1])**2)
            
            # Size similarity (how close to original size)
            size_ratio = min(det_size[0], self.original_size[0]) / max(det_size[0], self.original_size[0])
            size_ratio *= min(det_size[1], self.original_size[1]) / max(det_size[1], self.original_size[1])
            
            # Combined score: prefer close distance and similar size
            # Lower distance = better, higher size_ratio = better
            max_reasonable_distance = 300  # pixels
            if distance > max_reasonable_distance:
                continue  # Too far from last known position
            
            distance_score = 1.0 - (distance / max_reasonable_distance)
            score = (distance_score * 0.6) + (size_ratio * 0.4)
            
            if score > best_score:
                best_score = score
                best_match = (x1, y1, x2 - x1, y2 - y1)
        
        if best_match is not None and best_score > 0.3:
            return True, best_match
        return False, None
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bboxes."""
        x1_1, y1_1 = bbox1[0], bbox1[1]
        x2_1, y2_1 = bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - inter
        return inter / union
    
    def get_stats(self):
        """Return hybrid tracker statistics."""
        return {
            'yolo_checks': self.yolo_checks,
            'yolo_corrections': self.yolo_corrections,
            'frames_processed': self.frame_count
        }


class ValidationSuite:
    """Main validation suite for testing tracking algorithms under stress conditions."""
    
    def __init__(self, camera_index=1, output_dir="logs", tracker_type="CSRT"):
        """
        Initialize the validation suite.
        
        Args:
            camera_index (int): Camera device index (default: 1 for capture card)
            output_dir (str): Directory for storing logs and reports
            tracker_type (str): Tracker algorithm to use (CSRT, KCF, MOSSE, MIL)
        """
        self.camera_index = camera_index
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.tracker_type = tracker_type.upper()
        
        # Test data storage
        self.test_data = []
        self.start_time = None
        self.frame_count = 0
        
        # Tracking state
        self.tracker = None
        self.initial_center = None
        self.roi = None
        
        # Timestamp for unique log files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"test_log_{self.tracker_type}_{self.timestamp}.csv"
        
    def connect_camera(self):
        """
        Setup Phase: Connect to the capture card/camera.
        
        Returns:
            cv2.VideoCapture: Camera object if successful, None otherwise
        """
        print(f"\n{'='*60}")
        print("SETUP PHASE: Connecting to Camera")
        print(f"{'='*60}")
        print(f"Attempting to connect to camera index {self.camera_index}...")
        
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"❌ ERROR: Could not open camera at index {self.camera_index}")
            print("\nTroubleshooting:")
            print("  - Check if your HDMI capture card is connected")
            print("  - Try different camera indices: 0, 1, 2, etc.")
            print("  - Run with --debug flag to test with built-in webcam")
            return None
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"✅ Camera connected successfully!")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        
        return cap
    
    def calibration_phase(self, cap):
        """
        Calibration Phase: YOLO-based target selection.
        
        Uses YOLO to detect objects and lets user cycle through them,
        sorted by distance from screen center. Shows direction vector.
        
        Args:
            cap: OpenCV VideoCapture object
            
        Returns:
            bool: True if calibration successful, False otherwise
        """
        print(f"\n{'='*60}")
        print("CALIBRATION PHASE: YOLO-Based Target Selection")
        print(f"{'='*60}")
        
        # Check if YOLO is available for auto-selection
        if not YOLO_AVAILABLE:
            print("⚠️  YOLO not available, using manual selection...")
            return self._manual_calibration(cap)
        
        print("\nInstructions:")
        print("  - YOLO will detect objects in view")
        print("  - Objects are sorted by distance from center")
        print("  - Press N/RIGHT ARROW to cycle to next object")
        print("  - Press P/LEFT ARROW to cycle to previous object")  
        print("  - Press SPACE or ENTER to lock target (green → red)")
        print("  - Press ESC to cancel")
        print("\nLoading YOLO26 model...")
        
        # Load YOLO model for detection
        yolo_model = YOLO('yolo26n.pt')
        
        # Get frame dimensions for center calculation
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Could not read frame from camera")
            return False
        
        frame_height, frame_width = frame.shape[:2]
        screen_center = (frame_width // 2, frame_height // 2)
        
        print(f"   Screen center: {screen_center}")
        print("\nDetecting objects... Press keys to select.\n")
        
        selected_idx = 0
        locked = False
        locked_bbox = None
        locked_class = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection
            results = yolo_model(frame, verbose=False)
            
            # Extract detections and sort by distance from center
            detections = []
            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    if box.conf[0] < 0.25:
                        continue
                    
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    # Calculate distance from screen center
                    dist = np.sqrt((center[0] - screen_center[0])**2 + 
                                  (center[1] - screen_center[1])**2)
                    
                    class_id = int(box.cls[0])
                    class_name = yolo_model.names.get(class_id, 'unknown')
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                        'center': (int(center[0]), int(center[1])),
                        'distance': dist,
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': float(box.conf[0])
                    })
            
            # Sort by distance from center (closest first)
            detections.sort(key=lambda x: x['distance'])
            
            # Draw screen center crosshair
            cv2.line(frame, (screen_center[0] - 30, screen_center[1]), 
                    (screen_center[0] + 30, screen_center[1]), (255, 255, 255), 2)
            cv2.line(frame, (screen_center[0], screen_center[1] - 30), 
                    (screen_center[0], screen_center[1] + 30), (255, 255, 255), 2)
            
            # If locked (RED) - waiting for confirmation
            if locked and locked_bbox:
                x, y, w, h = locked_bbox
                obj_center = (x + w // 2, y + h // 2)
                
                # RED box (LOCKED - press space to start tracking)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
                
                # H/V direction vectors
                offset_x = obj_center[0] - screen_center[0]
                offset_y = obj_center[1] - screen_center[1]
                
                # Horizontal vector (cyan)
                h_end = (obj_center[0], screen_center[1])
                cv2.arrowedLine(frame, screen_center, h_end, (255, 255, 0), 3, tipLength=0.1)
                h_dir = "RIGHT" if offset_x > 0 else "LEFT"
                
                # Vertical vector (magenta)
                v_end = (screen_center[0], obj_center[1])
                cv2.arrowedLine(frame, screen_center, v_end, (255, 0, 255), 3, tipLength=0.1)
                v_dir = "DOWN" if offset_y > 0 else "UP"
                
                cv2.putText(frame, f"LOCKED: {locked_class} - SPACE to track", (x, y - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"H: {abs(offset_x)}px {h_dir}", (x, y - 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"V: {abs(offset_y)}px {v_dir}", (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Also draw other objects in green for reference
                for det in detections:
                    dx, dy, dw, dh = det['bbox']
                    cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), (0, 255, 0), 1)
            
            # Not locked - show all detections
            elif len(detections) > 0:
                for i, det in enumerate(detections):
                    x, y, w, h = det['bbox']
                    obj_center = det['center']
                    
                    if i == 0:
                        # CLOSEST TO CENTER: WHITE box (auto-selected)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
                        
                        # H/V direction vectors
                        offset_x = obj_center[0] - screen_center[0]
                        offset_y = obj_center[1] - screen_center[1]
                        
                        # Horizontal vector (cyan)
                        h_end = (obj_center[0], screen_center[1])
                        cv2.arrowedLine(frame, screen_center, h_end, (255, 255, 0), 3, tipLength=0.1)
                        h_dir = "RIGHT" if offset_x > 0 else "LEFT"
                        
                        # Vertical vector (magenta)
                        v_end = (screen_center[0], obj_center[1])
                        cv2.arrowedLine(frame, screen_center, v_end, (255, 0, 255), 3, tipLength=0.1)
                        v_dir = "DOWN" if offset_y > 0 else "UP"
                        
                        cv2.putText(frame, f"[CLOSEST] {det['class_name']}", 
                                   (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"H: {abs(offset_x)}px {h_dir}", 
                                   (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        cv2.putText(frame, f"V: {abs(offset_y)}px {v_dir}", 
                                   (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    else:
                        # Other detections: GREEN box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, det['class_name'], (x, y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "No objects detected - move camera", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Instructions overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, frame_height - 60), (450, frame_height - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            if locked:
                cv2.putText(frame, "SPACE: Start tracking | ESC: Cancel", 
                           (20, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "SPACE: Lock closest (WHITE) | ESC: Cancel", 
                           (20, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("YOLO Target Selection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Lock/Start tracking
            if key == ord(' ') or key == 13:  # Space or Enter
                if locked:
                    # Already locked - START TRACKING
                    cv2.destroyWindow("YOLO Target Selection")
                    self.roi = locked_bbox
                    self.initial_center = (locked_bbox[0] + locked_bbox[2] // 2,
                                          locked_bbox[1] + locked_bbox[3] // 2)
                    
                    # Initialize tracker
                    self.tracker = self._create_tracker()
                    self.tracker.init(frame, self.roi)
                    
                    print(f"✅ Target locked!")
                    print(f"   Tracker: {self.tracker_type}")
                    print(f"   Target: {locked_class}")
                    print(f"   ROI: {locked_bbox}")
                    print(f"   Initial Center: {self.initial_center}")
                    print(f"\nInitializing CSV log: {self.log_file}")
                    
                    self._initialize_csv()
                    return True
                    
                elif detections:
                    # Lock closest object (index 0)
                    det = detections[0]
                    locked = True
                    locked_bbox = det['bbox']
                    locked_class = det['class_name']
                    print(f"   Locked on: {locked_class} - press SPACE to start tracking")
            
            # Cancel
            elif key == 27:  # ESC
                cv2.destroyWindow("YOLO Target Selection")
                print("❌ Selection cancelled")
                return False
        
        return False
    
    def _manual_calibration(self, cap):
        """Fallback manual ROI selection."""
        print("\nInstructions:")
        print("  1. Draw a box around the object you want to track")
        print("  2. Press SPACE or ENTER to confirm")
        print("  3. Press ESC to cancel")
        
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Could not read frame from camera")
            return False
        
        self.roi = cv2.selectROI("Select Target to Track", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Target to Track")
        
        x, y, w, h = self.roi
        if w == 0 or h == 0:
            print("❌ ERROR: No valid ROI selected")
            return False
        
        self.initial_center = (x + w // 2, y + h // 2)
        self.tracker = self._create_tracker()
        self.tracker.init(frame, self.roi)
        
        print(f"✅ Target locked!")
        print(f"   Tracker: {self.tracker_type}")
        print(f"   ROI: ({x}, {y}, {w}, {h})")
        print(f"   Initial Center: {self.initial_center}")
        print(f"\nInitializing CSV log: {self.log_file}")
        
        self._initialize_csv()
        return True
    
    def _create_tracker(self):
        """Create a tracker instance based on selected type."""
        # Handle YOLO26 separately (detection-based, not correlation-based)
        if self.tracker_type == 'YOLO26':
            if not YOLO_AVAILABLE:
                print("❌ YOLO not available. Install with: pip install ultralytics")
                print("   Falling back to CSRT tracker.")
                self.tracker_type = 'CSRT'
            else:
                return YOLO26Tracker(model_size='n')
        
        # Handle HYBRID tracker (KCF + YOLO26)
        if self.tracker_type == 'HYBRID':
            if not YOLO_AVAILABLE:
                print("❌ YOLO not available for hybrid tracking.")
                print("   Falling back to KCF tracker.")
                self.tracker_type = 'KCF'
            else:
                return HybridTracker(yolo_interval=30, error_threshold=200)
        
        # Traditional OpenCV trackers with fallback handling
        # Some trackers may not be available depending on OpenCV build
        def try_create(name, create_func):
            try:
                return create_func()
            except AttributeError:
                print(f"⚠️  {name} tracker not available in this OpenCV build")
                return None
        
        # Try requested tracker first
        tracker_map = {
            'CSRT': lambda: try_create('CSRT', cv2.TrackerCSRT_create) if hasattr(cv2, 'TrackerCSRT_create') else None,
            'KCF': lambda: try_create('KCF', cv2.TrackerKCF_create) if hasattr(cv2, 'TrackerKCF_create') else None,
            'MOSSE': lambda: try_create('MOSSE', cv2.legacy.TrackerMOSSE_create) if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE_create') else None,
            'MIL': lambda: try_create('MIL', cv2.TrackerMIL_create) if hasattr(cv2, 'TrackerMIL_create') else None,
        }
        
        if self.tracker_type in tracker_map:
            tracker = tracker_map[self.tracker_type]()
            if tracker is not None:
                return tracker
        
        # Fallback chain: try each until one works
        print(f"⚠️  {self.tracker_type} not available, trying fallbacks...")
        fallback_order = ['MIL', 'CSRT', 'KCF']
        for fallback in fallback_order:
            if fallback in tracker_map:
                tracker = tracker_map[fallback]()
                if tracker is not None:
                    print(f"   Using {fallback} tracker instead")
                    self.tracker_type = fallback
                    return tracker
        
        raise RuntimeError("No compatible OpenCV tracker found!")
    
    def _initialize_csv(self):
        """Create CSV file with headers."""
        df = pd.DataFrame(columns=[
            'Time', 
            'Frame_ID', 
            'Target_X', 
            'Target_Y', 
            'Error_Distance', 
            'Signal_Quality_Metric',
            'Track_Status'
        ])
        df.to_csv(self.log_file, index=False)
        print(f"   CSV headers written")
    
    def calculate_signal_quality(self, frame):
        """
        Calculate signal quality using Laplacian variance (sharpness metric).
        
        Analog video degrades with static/noise - this manifests as blur.
        The Laplacian variance measures edge sharpness.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            float: Sharpness score (higher = better quality)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    def test_loop(self, cap):
        """
        Main Test Loop: Frame acquisition, tracking, validation, and logging.
        
        Args:
            cap: OpenCV VideoCapture object
        """
        print(f"\n{'='*60}")
        print("TEST LOOP: Validation in Progress")
        print(f"{'='*60}")
        print("\nControls:")
        print("  - Press 'Q' to stop the test and generate report")
        print("  - Perform stress tests by:")
        print("      * Moving the camera/object")
        print("      * Adding vibration (fan)")
        print("      * Degrading antenna signal")
        print("      * Changing lighting conditions")
        print("\nTest running...\n")
        
        self.start_time = datetime.now()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ WARNING: Lost camera feed")
                break
            
            self.frame_count += 1
            current_time = (datetime.now() - self.start_time).total_seconds()
            
            # Update tracker
            success, bbox = self.tracker.update(frame)
            
            if success:
                # Extract bounding box coordinates
                x, y, w, h = [int(v) for v in bbox]
                current_center = (x + w // 2, y + h // 2)
                
                # Calculate tracking error (drift from original position)
                error_distance = np.sqrt(
                    (current_center[0] - self.initial_center[0])**2 + 
                    (current_center[1] - self.initial_center[1])**2
                )
                
                # Draw tracking box (GREEN = tracking successful)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, current_center, 5, (0, 255, 0), -1)
                
                track_status = "TRACKING"
            else:
                # Tracker lost the target
                current_center = (0, 0)
                error_distance = 9999  # Sentinel value for "lost"
                track_status = "LOST"
                
                # Draw failure indicator (RED)
                cv2.putText(frame, "TRACKING LOST", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Calculate signal quality
            signal_quality = self.calculate_signal_quality(frame)
            
            # Log data
            self._log_frame_data(
                current_time, 
                self.frame_count, 
                current_center[0], 
                current_center[1], 
                error_distance, 
                signal_quality,
                track_status
            )
            
            # Display metrics on frame
            self._draw_metrics(frame, current_time, error_distance, signal_quality, track_status)
            
            # Get screen center
            frame_height, frame_width = frame.shape[:2]
            screen_center = (frame_width // 2, frame_height // 2)
            
            # Draw screen center crosshair
            cv2.line(frame, (screen_center[0] - 40, screen_center[1]), 
                    (screen_center[0] + 40, screen_center[1]), (255, 255, 255), 2)
            cv2.line(frame, (screen_center[0], screen_center[1] - 40), 
                    (screen_center[0], screen_center[1] + 40), (255, 255, 255), 2)
            cv2.putText(frame, "CENTER", (screen_center[0] + 45, screen_center[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            if success and current_center != (0, 0):
                # Calculate H and V offsets
                offset_x = current_center[0] - screen_center[0]
                offset_y = current_center[1] - screen_center[1]
                
                # Draw HORIZONTAL vector (cyan)
                h_end = (current_center[0], screen_center[1])
                cv2.arrowedLine(frame, (screen_center[0], screen_center[1]), h_end, 
                               (255, 255, 0), 3, tipLength=0.1)
                h_dir = "RIGHT -->" if offset_x > 0 else "<-- LEFT"
                cv2.putText(frame, f"H: {abs(offset_x)}px {h_dir}", 
                           (screen_center[0] + 50, screen_center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Draw VERTICAL vector (magenta)  
                v_end = (screen_center[0], current_center[1])
                cv2.arrowedLine(frame, (screen_center[0], screen_center[1]), v_end,
                               (255, 0, 255), 3, tipLength=0.1)
                v_dir = "DOWN v" if offset_y > 0 else "UP ^"
                cv2.putText(frame, f"V: {abs(offset_y)}px {v_dir}", 
                           (screen_center[0] + 50, screen_center[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                # Draw line from intersection to object (green)
                intersection = (current_center[0], screen_center[1])
                cv2.line(frame, intersection, current_center, (0, 255, 0), 2)
            
            # Display reference point (initial lock position)
            cv2.circle(frame, self.initial_center, 3, (255, 0, 0), -1)
            cv2.putText(frame, "ORIGIN", 
                       (self.initial_center[0] + 10, self.initial_center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Show frame
            cv2.imshow("Hardware Validation Suite - Live Feed", frame)
            
            # Check for quit command
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n✅ Test stopped by user")
                break
        
        print(f"\nTest completed: {self.frame_count} frames processed")
    
    def _log_frame_data(self, time, frame_id, x, y, error, quality, status):
        """Append data to CSV log file."""
        row = pd.DataFrame([{
            'Time': f"{time:.3f}",
            'Frame_ID': frame_id,
            'Target_X': x,
            'Target_Y': y,
            'Error_Distance': f"{error:.2f}",
            'Signal_Quality_Metric': f"{quality:.2f}",
            'Track_Status': status
        }])
        row.to_csv(self.log_file, mode='a', header=False, index=False)
    
    def _draw_metrics(self, frame, time, error, quality, status):
        """Draw real-time metrics overlay on frame."""
        # Background panel for better readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Metrics text
        y_offset = 30
        cv2.putText(frame, f"Time: {time:.1f}s", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        error_color = (0, 255, 0) if error < 50 else (0, 165, 255) if error < 100 else (0, 0, 255)
        cv2.putText(frame, f"Error: {error:.1f}px", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, error_color, 2)
        
        y_offset += 25
        cv2.putText(frame, f"Signal Quality: {quality:.0f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def generate_report(self):
        """
        Teardown Phase: Generate visualization and analysis report.
        """
        print(f"\n{'='*60}")
        print("GENERATING REPORT")
        print(f"{'='*60}")
        
        # Load the logged data
        df = pd.read_csv(self.log_file)
        
        if len(df) == 0:
            print("❌ No data to analyze")
            return
        
        print(f"Analyzing {len(df)} frames...")
        
        # Create reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Generate visualization
        report_path = reports_dir / f"validation_report_{self.timestamp}.png"
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Hardware Validation Suite - Test Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Tracking Error over Time
        ax1.plot(df['Time'], df['Error_Distance'], 'b-', linewidth=2, label='Tracking Error')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Error Distance (pixels)', fontsize=12)
        ax1.set_title('Tracking Error vs. Time', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Highlight tracking failures
        lost_frames = df[df['Track_Status'] == 'LOST']
        if len(lost_frames) > 0:
            ax1.scatter(lost_frames['Time'], lost_frames['Error_Distance'], 
                       color='red', s=100, marker='x', label='Tracking Lost', zorder=5)
        
        # Plot 2: Signal Quality over Time
        ax2.plot(df['Time'], df['Signal_Quality_Metric'], 'g-', linewidth=2, label='Signal Quality')
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Sharpness Score', fontsize=12)
        ax2.set_title('Signal Quality vs. Time (Laplacian Variance)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add threshold line
        ax2.axhline(y=500, color='r', linestyle='--', alpha=0.7, label='Critical Threshold')
        
        # Plot 3: Correlation (Error vs Signal Quality)
        valid_data = df[df['Track_Status'] == 'TRACKING']
        ax3.scatter(valid_data['Signal_Quality_Metric'], valid_data['Error_Distance'], 
                   alpha=0.5, s=20)
        ax3.set_xlabel('Signal Quality (Sharpness)', fontsize=12)
        ax3.set_ylabel('Tracking Error (pixels)', fontsize=12)
        ax3.set_title('Error vs. Signal Quality Correlation', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add trendline
        if len(valid_data) > 1:
            z = np.polyfit(valid_data['Signal_Quality_Metric'], valid_data['Error_Distance'], 1)
            p = np.poly1d(z)
            ax3.plot(valid_data['Signal_Quality_Metric'], 
                    p(valid_data['Signal_Quality_Metric']), 
                    "r--", alpha=0.8, linewidth=2, label='Trend')
            ax3.legend()
        
        plt.tight_layout()
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f"✅ Report saved: {report_path}")
        
        # Print summary statistics
        self._print_summary(df)
        
        # Show the plot
        plt.show()
    
    def _print_summary(self, df):
        """Print test summary statistics."""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        
        total_frames = len(df)
        tracking_frames = len(df[df['Track_Status'] == 'TRACKING'])
        lost_frames = total_frames - tracking_frames
        
        print(f"Total Frames: {total_frames}")
        print(f"Tracking Success: {tracking_frames} ({tracking_frames/total_frames*100:.1f}%)")
        print(f"Tracking Lost: {lost_frames} ({lost_frames/total_frames*100:.1f}%)")
        
        if tracking_frames > 0:
            valid_data = df[df['Track_Status'] == 'TRACKING']
            avg_error = valid_data['Error_Distance'].mean()
            max_error = valid_data['Error_Distance'].max()
            avg_quality = valid_data['Signal_Quality_Metric'].mean()
            
            print(f"\nTracking Performance:")
            print(f"  Average Error: {avg_error:.2f} pixels")
            print(f"  Maximum Error: {max_error:.2f} pixels")
            print(f"  Average Signal Quality: {avg_quality:.2f}")
            
            # The "WOW" factor - correlation analysis
            low_quality_frames = valid_data[valid_data['Signal_Quality_Metric'] < 500]
            if len(low_quality_frames) > 0:
                low_quality_error = low_quality_frames['Error_Distance'].mean()
                high_quality_frames = valid_data[valid_data['Signal_Quality_Metric'] >= 500]
                
                if len(high_quality_frames) > 0:
                    high_quality_error = high_quality_frames['Error_Distance'].mean()
                    error_increase = ((low_quality_error - high_quality_error) / high_quality_error) * 100
                    
                    print(f"\n{'🎯 KEY FINDING (The Interview Story)':^60}")
                    print(f"{'='*60}")
                    print(f"When signal quality dropped below 500:")
                    print(f"  - Low Quality Error: {low_quality_error:.2f}px")
                    print(f"  - High Quality Error: {high_quality_error:.2f}px")
                    print(f"  - Error Increase: {error_increase:+.1f}%")
                    print(f"\n💡 This proves the algorithm needs better noise filtering!")
        
        print(f"\n{'='*60}")
        print(f"Log file: {self.log_file}")
        print(f"{'='*60}\n")
    
    def run(self):
        """Execute the complete validation suite workflow."""
        print("\n" + "="*60)
        print(" "*10 + "HARDWARE VALIDATION SUITE")
        print(" "*8 + "Drone FPV Tracking System Test")
        print("="*60)
        
        # Setup Phase
        cap = self.connect_camera()
        if cap is None:
            return False
        
        try:
            # Calibration Phase
            if not self.calibration_phase(cap):
                return False
            
            # Test Loop
            self.test_loop(cap)
            
            # Teardown & Report Generation
            self.generate_report()
            
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
        
        return True


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Hardware Validation Suite for FPV Drone Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use capture card (default camera index 1)
  python validation_suite.py
  
  # Use built-in webcam for testing
  python validation_suite.py --camera 0 --debug
  
  # Specify custom camera index
  python validation_suite.py --camera 2
        """
    )
    
    parser.add_argument('--camera', type=int, default=1,
                       help='Camera device index (default: 1 for capture card)')
    parser.add_argument('--output', type=str, default='logs',
                       help='Output directory for logs (default: logs)')
    parser.add_argument('--tracker', type=str, default='CSRT',
                       choices=['CSRT', 'KCF', 'MOSSE', 'MIL', 'YOLO26', 'HYBRID'],
                       help='Tracker: CSRT, KCF, MOSSE, MIL, YOLO26 (AI), HYBRID (KCF+YOLO26)')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode - use webcam instead of capture card')
    
    args = parser.parse_args()
    
    # Override camera index in debug mode
    if args.debug:
        print("🔧 DEBUG MODE: Using camera index 0 (built-in webcam)")
        args.camera = 0
    
    # Run the suite
    suite = ValidationSuite(camera_index=args.camera, output_dir=args.output, tracker_type=args.tracker)
    success = suite.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
