import numpy as np
import tensorflow as tf
from PIL import Image
import os
import math

class Physical_Adaptor():
    def __init__(self,args, content_seg,content_image, input_image,content_width, content_height):
        tf_content_seg = tf.constant(content_seg)
        tf_reverse_content_seg = tf.constant(1- content_seg)
        masked_content = tf.multiply(tf.constant(content_image),tf_reverse_content_seg)
        masked_style =tf.multiply(input_image,tf_content_seg)
        self.transformed_image = tf.add(masked_content ,masked_style)
        self.background = tf.placeholder(tf.float32, (None,content_height,content_width,3))
        self.img_with_bg = self.img_random_overlay(self.background,tf_content_seg,content_width, self.transformed_image)
        self.img_with_bg = tf.clip_by_value(self.img_with_bg, 0.0 ,255.0)
        self.resized_img = tf.image.resize_images(self.img_with_bg, (224,224))
        self.bg_path = args.background_path


    def _transform_vector(self, width, x_shift, y_shift, im_scale, rot_in_degrees):
        """
            If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1],
            then it maps the output point (x, y) to a transformed input point
            (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k),
            where k = c0 x + c1 y + 1.
            The transforms are inverted compared to the transform mapping input points to output points.
            """
        rot = float(rot_in_degrees) / 90. * (math.pi / 2)
        # Standard rotation matrix
        # (use negative rot because tf.contrib.image.transform will do the inverse)
        rot_matrix = np.array([[math.cos(-rot), -math.sin(-rot)], [math.sin(-rot), math.cos(-rot)]])

        # Scale it
        # (use inverse scale because tf.contrib.image.transform will do the inverse)
        inv_scale = 1. / im_scale
        xform_matrix = rot_matrix * inv_scale
        a0, a1 = xform_matrix[0]
        b0, b1 = xform_matrix[1]

        # At this point, the image will have been rotated around the top left corner,
        # rather than around the center of the image.
        #
        # To fix this, we will see where the center of the image got sent by our transform,
        # and then undo that as part of the translation we apply.
        x_origin = float(width) / 2
        y_origin = float(width) / 2

        x_origin_shifted, y_origin_shifted = np.matmul(xform_matrix, np.array([x_origin, y_origin]), )

        x_origin_delta = x_origin - x_origin_shifted
        y_origin_delta = y_origin - y_origin_shifted

        # Combine our desired shifts with the rotation-induced undesirable shift
        a2 = x_origin_delta - (x_shift / (2 * im_scale))
        b2 = y_origin_delta - (y_shift / (2 * im_scale))

        # Return these values in the order that tf.contrib.image.transform expects
        return np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32)


    def _random_transformation(self, min_scale, width, max_rotation):
        """Random resize and rotation.
        
        Arguments:
            min_scale {float32} -- Minimize scale of adv compared to background (supposed the scale of background as 1)
            width {float32} -- Width of adv.
            max_rotation {float32} -- Max rotation degree of adv.
        
        """
        im_scale = np.random.uniform(low=min_scale, high=0.6)

        padding_after_scaling = (1 - im_scale) * width
        x_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
        y_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)

        rot = np.random.uniform(-max_rotation, max_rotation)

        return self._transform_vector(width,
                                x_shift=x_delta,
                                y_shift=y_delta,
                                im_scale=im_scale,
                                rot_in_degrees=rot)


    def select_random_background(self, content_height, content_width):
        """"
        The function return a random background from specified path.
        """
        # bg_dic = {'t-shirt':'./background/t-shirt','traffic': './physical-attack-data/background/traffic_bg','banana':'./physical-attack-data/background/banana'}
        files = os.listdir(self.bg_path)
        rand_num = np.random.randint(0,len(files ))
        file_name = os.path.join(self.bg_path,files[rand_num])
        bg = np.array(Image.open(file_name).convert("RGB").resize((content_height,content_width)), dtype=np.float32)
        bg = bg
        bg = np.expand_dims(bg,0)
        return bg

    def img_random_overlay(self, bg, img_mask,width,adv_img,min_scale=0.4, max_rotation = 25):
        """adv with background
        
        Arguments:
            bg {tensor} -- selected background
            img_mask {tensor} -- Rotation and resize
            width {float32} -- Width of img
            adv {tensor} -- adversarial_img
        
        Keyword Arguments:
            min_scale {float} -- [description] (default: {0.4})
            max_rotation {int} -- [description] (default: {25})
        
        Returns:
            [type] -- [description]
        """
        bg = tf.squeeze(bg,[0])
        adv_img = tf.squeeze(adv_img,[0])
        random_xform_vector = tf.py_func(self._random_transformation, [min_scale, width, max_rotation], tf.float32)
        random_xform_vector.set_shape([8])
        output = tf.contrib.image.transform(adv_img, random_xform_vector, "BILINEAR")
        input_mask = tf.contrib.image.transform(img_mask , random_xform_vector, "BILINEAR")
        background_mask = 1-input_mask
        input_with_background = tf.add(tf.multiply(background_mask, bg),  tf.multiply(input_mask , output))

        #For simulatinng lightnness change
        color_shift_input = input_with_background +  input_with_background * tf.constant(np.random.uniform(-0.3,0.3))
        img_with_bg = tf.expand_dims(color_shift_input,0)
        return img_with_bg