import os

image_folder = './Show-o'

for filename in os.listdir(image_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        new_filename = filename.replace('swapped', '0')
        src = os.path.join(image_folder, filename)
        dst = os.path.join(image_folder, new_filename)
        os.rename(src, dst)
        print(f'Renamed: {filename} -> {new_filename}')