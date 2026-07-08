import cv2
import numpy as np

# Load your original setup image
# Replace with the exact path to your original image file
img_path = "duckweed_images/DSCF0039.jpg" 
img = cv2.imread(img_path)

if img is None:
    print("Could not open image. Please check the path.")
    exit()

height, width, _ = img.shape

# Window setup
cv2.namedWindow("Tune ROI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tune ROI", 800, 600)

def nothing(x):
    pass

# Trackbars for Left Tub (X, Y, Radius)
cv2.createTrackbar("L_X", "Tune ROI", int(width * 0.28), width, nothing)
cv2.createTrackbar("L_Y", "Tune ROI", int(height * 0.38), height, nothing)
cv2.createTrackbar("L_R", "Tune ROI", int(width * 0.22), int(width/2), nothing)

# Trackbars for Right Tub (X, Y, Radius)
cv2.createTrackbar("R_X", "Tune ROI", int(width * 0.68), width, nothing)
cv2.createTrackbar("R_Y", "Tune ROI", int(height * 0.38), height, nothing)
cv2.createTrackbar("R_R", "Tune ROI", int(width * 0.22), int(width/2), nothing)

print("Adjust sliders until circles cover the tubs perfectly. Press 'q' to print values and exit.")

while True:
    clone = img.copy()
    
    # Get current positions of trackbars
    lx = cv2.getTrackbarPos("L_X", "Tune ROI")
    ly = cv2.getTrackbarPos("L_Y", "Tune ROI")
    lr = cv2.getTrackbarPos("L_R", "Tune ROI")
    
    rx = cv2.getTrackbarPos("R_X", "Tune ROI")
    ry = cv2.getTrackbarPos("R_Y", "Tune ROI")
    rr = cv2.getTrackbarPos("R_R", "Tune ROI")
    
    # Draw transparent overlays to check alignment
    overlay = clone.copy()
    cv2.circle(overlay, (lx, ly), lr, (255, 0, 0), -1)
    cv2.circle(overlay, (rx, ry), rr, (255, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, clone, 0.6, 0, clone)
    
    cv2.imshow("Tune ROI", clone)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n--- Copy these coordinates into your main segmentation script: ---")
        print(f"left_tub = ({lx}, {ly}, {lr})")
        print(f"right_tub = ({rx}, {ry}, {rr})")
        break

cv2.destroyAllWindows()