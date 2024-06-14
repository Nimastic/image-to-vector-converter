import cv2
import numpy as np
import matplotlib.pyplot as plt

def vectorize_image(image_path, output_path, scale_factor=2):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty image to draw contours
    vectorized_image = np.zeros_like(image)
    
    # Draw contours on the empty image
    cv2.drawContours(vectorized_image, contours, -1, (0, 255, 0), 1)
    
    # Resize the original and vectorized images
    original_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    vectorized_resized = cv2.resize(vectorized_image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    
    # Save the vectorized image
    cv2.imwrite(output_path, vectorized_resized)
    
    # Display the original and vectorized images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 2, 2)
    plt.title('Vectorized Image')
    plt.imshow(cv2.cvtColor(vectorized_resized, cv2.COLOR_BGR2RGB))
    plt.show()

# Usage
image_path = '/mnt/data/image.png'
output_path = '/mnt/data/vectorized_image.png'
vectorize_image(image_path, output_path, scale_factor=2)
