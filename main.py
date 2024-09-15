import cv2
import numpy as np

# Load the image (replace 'input_image.png' with your image file)
img = cv2.imread('C:/Users/frank/Projects/brush_tool/Blunge Output Images/00.png', cv2.IMREAD_UNCHANGED)


# Check if image has an alpha channel; if not, add one
if img.shape[2] == 3:
    # Add an alpha channel (fully opaque)
    alpha_channel = np.ones(img.shape[:2], dtype=img.dtype) * 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img[:, :, 3] = alpha_channel

# Create a copy of the image for display purposes
img_display = img.copy()

# Create a mask where we'll draw the brush strokes
mask = np.zeros(img.shape[:2], dtype=np.uint8)

# Variables to store drawing state
drawing = False  # True if the mouse is pressed
brush_size = 20  # Size of the brush

def draw_brush(event, x, y, flags, param):
    global drawing, mask, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(mask, (x, y), brush_size, 255, -1)  # Draw on the mask
        cv2.circle(img_display, (x, y), brush_size, (0, 0, 255, 255), -1)  # Visual feedback on the image
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x, y), brush_size, 255, -1)
            cv2.circle(img_display, (x, y), brush_size, (0, 0, 255, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(mask, (x, y), brush_size, 255, -1)
        cv2.circle(img_display, (x, y), brush_size, (0, 0, 255, 255), -1)

# Set up the window and bind the draw_brush function to mouse events
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_brush)

print("Instructions:")
print(" - Draw over the image to select areas to make transparent.")
print(" - Press 's' to apply transparency and save the image.")
print(" - Press 'q' to exit without saving.")

while True:
    cv2.imshow('Image', img_display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Apply transparency where the mask is white
        img[:, :, 3][mask == 255] = 0  # Set alpha to 0 (transparent)
        cv2.imwrite('output_image.png', img)
        print("Image saved as 'output_image.png'.")
        break
    elif key == ord('q'):
        print("Exiting without saving.")
        break

cv2.destroyAllWindows()
