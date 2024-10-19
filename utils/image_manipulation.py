import tensorflow as tf

IMG_SIZE = (300, 300)

def resize_and_pad(image, label=None):
    image = tf.image.resize_with_pad(image, target_height=IMG_SIZE[0], target_width=IMG_SIZE[1])

    if label is not None:
        return image, label
    else:
        return image
    

def swap_labels(image, label):
    label = tf.where(label == 0, 1, 0)
    
    return image, label