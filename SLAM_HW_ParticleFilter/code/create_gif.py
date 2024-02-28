'''
code to create gif from PNGs
finished by Tianqi Yu (tianqiyu@andrew.cmu.edu), 2024
'''
import imageio
import os

# input 
image_folder = '../code/results'
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
print('successread')
# output
output_path = '../code/results'
gif_name = os.path.join(output_path, 'output_1.gif')


with imageio.get_writer(gif_name, mode='I') as writer:
    for image in images:
        image_path = os.path.join(image_folder, image)
        writer.append_data(imageio.imread(image_path))
