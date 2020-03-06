
import numpy as np
import os
import tensorflow as tf
from PIL import Image


def load_imgs_from_path(args, target_size =(400,400)):

    content_image = np.array(Image.open(args.content_image_path).convert("RGB").resize(target_size), dtype=np.float32)
    content_width, content_height = content_image.shape[1], content_image.shape[0]
    content_image = content_image.reshape((1, content_height, content_width, 3)).astype(np.float32)

    style_image = np.array(Image.open(args.style_image_path).convert("RGB"), dtype=np.float32)
    style_width, style_height = style_image.shape[1], style_image.shape[0]
    style_image = style_image.reshape((1, style_height, style_width, 3)).astype(np.float32)
    content_seg = np.array(Image.open(args.content_seg_path).convert("RGB").resize((content_width, content_height), resample=Image.BILINEAR), dtype=np.float32) // 245.0
    style_seg = np.array(Image.open(args.style_seg_path).convert("RGB").resize((style_width, style_height), resample=Image.BILINEAR), dtype=np.float32) // 245.0

    return content_image,style_image,content_seg,style_seg
    
def load_seg(content_seg, style_seg ):
    color_codes = ['UnAttack', 'Attack']
    content_shape = [content_seg.shape[1],content_seg.shape[0]] 
    style_shape = [style_seg.shape[1],style_seg.shape[0]]
    with tf.name_scope('segmentation'):
        
        def _extract_mask(seg, color_str):
            h, w, c = np.shape(seg)
            if color_str == "UnAttack":
                mask_r = (seg[:, :, 0] < 0.5).astype(np.uint8)
                mask_g = (seg[:, :, 1] < 0.5).astype(np.uint8)
                mask_b = (seg[:, :, 2] < 0.5).astype(np.uint8)
            elif color_str == "Attack":
                mask_r = (seg[:, :, 0] > 0.8).astype(np.uint8)
                mask_g = (seg[:, :, 1] > 0.8).astype(np.uint8)
                mask_b = (seg[:, :, 2] > 0.8).astype(np.uint8)
          
            return np.multiply(np.multiply(mask_r, mask_g), mask_b).astype(np.float32)
        
        color_content_masks = []
        color_style_masks = []
        for i in range(len(color_codes)):
            color_content_masks.append(tf.expand_dims(tf.expand_dims(tf.constant(_extract_mask(content_seg, color_codes[i])), 0), -1))
            color_style_masks.append(tf.expand_dims(tf.expand_dims(tf.constant(_extract_mask(style_seg, color_codes[i])), 0), -1))

    return color_content_masks, color_style_masks

def gram_matrix(activations):
    height = tf.shape(activations)[1]
    width = tf.shape(activations)[2]
    num_channels = tf.shape(activations)[3]
    gram_matrix = tf.transpose(activations, [0, 3, 1, 2])
    gram_matrix = tf.reshape(gram_matrix, [num_channels, width * height])
    gram_matrix = tf.matmul(gram_matrix, gram_matrix, transpose_b=True)
    return gram_matrix

def save_result(img_, str_):
    result = Image.fromarray(np.uint8(np.clip(img_, 0, 255.0)))
    result.save(str_)


def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return pred[0], prob[pred[0]]


def intra_class(dir_path, img_path):
    list_of_imgs = os.listdir(dir_path)
    random_id = np.random.randint(0,len(list_of_imgs))
    selceted_img_path = os.path.join(dir_path, list_of_imgs[random_id])
    while selceted_img_path == img_path:
        random_id = np.random.randint(0,len(list_of_imgs))
        selceted_img_path = os.path.join(dir_path)
    return selceted_img_path

