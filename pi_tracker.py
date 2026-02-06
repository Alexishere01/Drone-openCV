import cv2
import sys
import time
import json
import csv
import math
from datetime import datetime
from ultralytics import YOLO

def main():
    print("🍓 Pi Tracker: Initializing...")
    
    # 1. Initialize Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"tracker_log_{timestamp}.csv"
    
    with open(log_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Frame_ID", "Target_Class", "Center_X", "Center_Y", "Target_X", "Target_Y", "Vector_X", "Vector_Y"])
    
    print(f"📝 Logging to: {log_filename}")

    # 2. Initialize Camera
    camera_index = 0
    if len(sys.argv) > 1:
        camera_index = int(sys.argv[1])
        
    cap = cv2.VideoCapture(camera_index)
    
    # Force standard resolution (720p is good balance for Pi 5 + YOLO)
    width, height = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {camera_index}")
        return

    # 3. Load YOLO Model (Nano version for speed)
    print("🧠 Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt') 

    # 4. Setup Display
    window_name = "Pi Tracker Output"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    screen_center = (width // 2, height // 2)
    print(f"🎯 Screen Center: {screen_center}")
    print("🚀 Tracking started! Press 'q' to exit, 'SPACE' to lock/unlock.")

    frame_id = 0
    locked_track_id = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Frame capture failed")
                break
            
            frame_id += 1
            
            # --- Detection & Tracking Logic ---
            # persists=True is crucial for ID tracking across frames
            results = model.track(frame, persist=True, verbose=False)
            
            # 1. Parse all detections into a useful list
            detections = []
            if results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes
                track_ids = boxes.id.int().cpu().tolist()
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    dist = math.sqrt((cx - screen_center[0])**2 + (cy - screen_center[1])**2)
                    
                    detections.append({
                        "id": track_id,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "center": (cx, cy),
                        "dist": dist,
                        "name": model.names[int(box.cls[0])],
                        "conf": float(box.conf[0])
                    })

            # 2. Logic Controller
            target_obj = None
            
            if locked_track_id is not None:
                # --- LOCKED MODE: Find our specific ID ---
                target_obj = next((d for d in detections if d["id"] == locked_track_id), None)
                
                if target_obj:
                    # Found our target!
                    color = (0, 255, 0) # Green
                    status_text = f"LOCKED: {target_obj['name']} #{locked_track_id}"
                else:
                    # Lost target this frame
                    color = (0, 0, 255) # Red
                    status_text = f"SEARCHING: {locked_track_id}..."
                    
            else:
                # --- SEARCH MODE: Find closest to center ---
                if detections:
                    # Sort by distance to center
                    detections.sort(key=lambda x: x["dist"])
                    target_obj = detections[0]
                    color = (255, 255, 255) # White (Candidate)
                    status_text = f"READY: {target_obj['name']} #{target_obj['id']}"
                else:
                    status_text = "NO OBJECTS"
                    color = (200, 200, 200)

            # 3. Draw & Output
            # Draw crosshair
            cv2.line(frame, (screen_center[0]-20, screen_center[1]), (screen_center[0]+20, screen_center[1]), (100, 100, 100), 2)
            cv2.line(frame, (screen_center[0], screen_center[1]-20), (screen_center[0], screen_center[1]+20), (100, 100, 100), 2)

            data_packet = {
                "frame": frame_id,
                "status": "search" if locked_track_id is None else "locked",
                "target_id": locked_track_id,
                "dx": 0,
                "dy": 0
            }

            if target_obj:
                x1, y1, x2, y2 = target_obj['bbox']
                tx, ty = target_obj['center']
                
                # If we are locked OR this is the candidate, calculate vector
                dx = tx - screen_center[0]
                dy = ty - screen_center[1]
                
                if locked_track_id is not None and target_obj['id'] == locked_track_id:
                    data_packet["dx"] = dx
                    data_packet["dy"] = dy
                    
                    # Draw strong relationship line
                    cv2.arrowedLine(frame, screen_center, (tx, ty), color, 4)
                    
                    # CSV Log (Only when locked and valid)
                    with open(log_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.now().isoformat(), frame_id, target_obj["name"],
                            screen_center[0], screen_center[1], tx, ty, dx, dy
                        ])
                elif locked_track_id is None:
                    # Candidate styling (dotted line or thinner)
                    cv2.line(frame, screen_center, (tx, ty), color, 1)

                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"#{target_obj['id']} {target_obj['name']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # HUD
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, "[SPACE] Lock/Unlock | [Q] Quit", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Print JSON
            print(json.dumps(data_packet)) 

            cv2.imshow(window_name, frame)
            
            # Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if locked_track_id is None:
                    if target_obj:
                        locked_track_id = target_obj['id']
                        print(f"🔒 LOCKED onto ID {locked_track_id}")
                else:
                    print(f"🔓 UNLOCKED ID {locked_track_id}")
                    locked_track_id = None
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Finished. Log saved to {log_filename}")

if __name__ == "__main__":
    main()
