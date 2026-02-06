import cv2
import sys

def main():
    """
    Simple script to display the camera feed in full screen.
    Run this on the Raspberry Pi.
    """
    print("Initializing camera...")
    
    # Try index 0 first, then -1 (any available)
    camera_index = 0
    if len(sys.argv) > 1:
        camera_index = int(sys.argv[1])
        
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}")
        print("Try: libcamera-hello to verify camera works outside of Python")
        return

    # Set resolution to 1080p (or highest available)
    # The capture card will likely pick this up as 1080p60 or 1080p30
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Create a full screen window
    window_name = "Pi Camera Output"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"Camera started. Displaying to {window_name}")
    print("Press 'q' or ESC to exit")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
