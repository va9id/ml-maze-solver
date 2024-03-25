import cv2
from constants import IMAGE_SIZE


def resize_image(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    '''
    Resizes image while preserving aspect ratios

    Parameters:
    image: the image to resize

    Returns:
    image: the resized image
    '''
    resized_width = IMAGE_SIZE[1]
    resized_height = IMAGE_SIZE[0]
    original_height, original_width = image.shape[:2]

    # Calculate the scaling factor for both dimensions
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    scale = min(scale_x, scale_y)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    return cv2.resize(image, (new_width, new_height))