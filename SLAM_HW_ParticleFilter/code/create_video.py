'''
code to create video from PNGs
finished by Tianqi Yu (tianqiyu@andrew.cmu.edu), 2024
'''

import cv2
import os

# input 
image_folder = '../code/results'
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

# output
output_path = '../code/results'
video_name = os.path.join(output_path, 'output_1.avi')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
# Set the desired width and height (e.g., doubling the original resolution)
width *= 2
height *= 2
video = cv2.VideoWriter(video_name, fourcc, 5, (width,height))

for image in images:
    img = cv2.imread(os.path.join(image_folder, image))
    # Resize the image to the desired resolution
    img = cv2.resize(img, (width, height))
    video.write(img)

cv2.destroyAllWindows()
video.release()
