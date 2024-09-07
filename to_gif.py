import os
from PIL import Image


def create_gif_from_images(input_folder: str, output_gif: str,
                           duration: int = 100) -> None:
    """
    Create a GIF from all .jpg images in a folder, sorted by their numeric filenames.

    :param input_folder: Folder containing the .jpg images.
    :param output_gif: Path to save the output .gif file.
    :param duration: Duration (in milliseconds) for each frame in the GIF. Default is 500ms.
    """
    # Get all jpg images from the input folder and sort them by numeric order
    image_filenames = [f for f in os.listdir(input_folder) if
                       f.endswith('.jpg')]
    image_filenames.sort(key=lambda x: int(
        os.path.splitext(x)[0]))  # Sort by numeric value in filename

    # Load images
    images = [Image.open(os.path.join(input_folder, filename)) for filename in
              image_filenames]

    # Save as GIF
    if images:
        images[0].save(output_gif, save_all=True, append_images=images[1:],
                       duration=duration, loop=0)
        print(f"GIF saved to {output_gif}")
    else:
        print("No images found in the folder.")


# Example usage:
input_folder = 'yolo_world_sam'  # Replace with your folder path
output_gif = f'{input_folder}.gif'  # Replace with your desired output GIF path

create_gif_from_images(input_folder, output_gif)
