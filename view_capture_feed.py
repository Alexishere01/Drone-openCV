import cv2
import sys

def list_cameras():
    """
    Test the first 10 indexes to see what cameras are available.
    """
    print("Listing available cameras...")
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Index {i}: Available")
            cap.release()
    print("Done listing.")

def main():
    """
    Script to view HDMI Capture Card input on your Laptop.
    """
    if "--list" in sys.argv:
        list_cameras()
        return

    # Default to index 1 (common for external capture cards on Mac)
    # If using built-in webcam, it's usually 0.
    camera_index = 1
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        camera_index = int(sys.argv[1])

    print(f"Opening camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera index {camera_index}.")
        print("Try running with --list to see available cameras.")
        return

    # Set resolution to 1080p for better quality if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    window_name = f"Capture Card Feed (Index {camera_index})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Displaying feed from index {camera_index}.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame from capture card.")
            break

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
