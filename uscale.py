import cv2
import numpy as np
import matplotlib.pyplot as plt

def upscale_image(image_path, output_path, scale_factor=2):
    # Load the image
    image = cv2.imread(image_path)
    
    # Resize the original image
    original_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    
    # Save the resized image
    cv2.imwrite(output_path, original_resized)
    
    # Display the original and resized images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 2, 2)
    plt.title('Upscaled Image')
    plt.imshow(cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB))
    plt.show()


# Usage
image_path = "C:/Users/jerie/Downloads/EWV.png"
output_path = "C:/Users/jerie/Downloads/vectorized_EWV.png"
upscale_image(image_path, output_path, scale_factor=16)
