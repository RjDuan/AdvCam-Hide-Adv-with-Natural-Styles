# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
import tensorflow as tf
from vgg19.vgg import Vgg19
from PIL import Image
import os
import math
from utils import *
from physical_adaption_utils import Physical_Adaptor

def content_loss(const_layer, var_layer, weight):
    return tf.reduce_mean(tf.squared_difference(const_layer, var_layer)) * weight


def style_loss(CNN_structure, const_layers, content_const_layers, var_layers, content_segs, style_segs, weight):
    with tf.name_scope('style_loss'):
        loss_styles = []
        layer_count = float(len(const_layers))
        layer_index = 0
        
        _, content_seg_height, content_seg_width, _ = content_segs[0].get_shape().as_list()
        _, style_seg_height, style_seg_width, _ = style_segs[0].get_shape().as_list()
        for layer_name in CNN_structure:
            layer_name = layer_name[layer_name.find("/") + 1:]
            with tf.name_scope('style_loss_layer'):
                # downsampling segmentation
                if "pool" in layer_name:
                    content_seg_width, content_seg_height = int(math.ceil(content_seg_width / 2)), int(math.ceil(content_seg_height / 2))
                    style_seg_width, style_seg_height = int(math.ceil(style_seg_width / 2)), int(math.ceil(style_seg_height / 2))
                    
                    for i in range(len(content_segs)):
                        content_segs[i] =tf.image.resize_bilinear(content_segs[i],tf.constant((content_seg_height, content_seg_width)))
                        style_segs[i] = tf.image.resize_bilinear(style_segs[i],tf.constant((style_seg_height, style_seg_width)))
            
                elif "conv" in layer_name:
                    for i in range(len(content_segs)):
                        content_segs[i] = tf.nn.avg_pool(tf.pad(content_segs[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')
                        style_segs[i] = tf.nn.avg_pool(tf.pad(style_segs[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"),
                                                                                        ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')
                
                if layer_name == var_layers[layer_index].name[var_layers[layer_index].name.find("/") + 1:]:
                    print("Setting up style layer: <{}>".format(layer_name))
                    const_layer = const_layers[layer_index]
                    content_const_layer = content_const_layers[layer_index]
                    var_layer = var_layers[layer_index]
                    
                    layer_index = layer_index + 1
                    
                    layer_style_loss = 0.0
                    for content_seg, style_seg in zip(content_segs, style_segs):
                        gram_matrix_var = gram_matrix(tf.multiply(var_layer, content_seg))
                        content_mask_mean = tf.reduce_mean(content_seg)
                        gram_matrix_var = tf.cond(tf.greater(content_mask_mean, 0.),
                                                  lambda:gram_matrix_var / (tf.to_float(tf.size(var_layer)) * content_mask_mean),
                                                  lambda: gram_matrix_var)
                        cur_style_mask_mean = tf.reduce_mean(style_seg)
                        style_mask_mean = tf.cond(tf.logical_and(tf.greater(content_mask_mean, 0.), (tf.equal(0., cur_style_mask_mean))),
                                                  lambda: tf.reduce_mean(content_seg),
                                                  lambda: tf.reduce_mean(style_seg))
                        cur_const_layer = tf.cond(tf.logical_and(tf.greater(content_mask_mean, 0.), (tf.equal(0., cur_style_mask_mean))),
                                                  lambda: content_const_layer,
                                                  lambda: const_layer)
                        gram_matrix_const = tf.cond(tf.logical_and(tf.greater(content_mask_mean, 0.), (tf.equal(0., cur_style_mask_mean))),
                                                    lambda: gram_matrix(tf.multiply(content_const_layer, content_seg)),
                                                    lambda: gram_matrix(tf.multiply(const_layer, style_seg)))
                        gram_matrix_const = tf.cond(tf.greater(style_mask_mean, 0.),
                                                    lambda: gram_matrix_const / (tf.to_float(tf.size(cur_const_layer)) * style_mask_mean),
                                                    lambda: gram_matrix_const )
                        diff_style_sum = tf.reduce_mean(tf.squared_difference(gram_matrix_const, gram_matrix_var)) * content_mask_mean
                        layer_style_loss += diff_style_sum
                    
                    loss_styles.append(layer_style_loss * weight)
    return loss_styles


def total_variation_loss(output, weight):
    tv_loss = tf.reduce_sum(
        (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) * (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) + \
        (output[:, :-1, :-1, :] - output[:, 1:, :-1, :]) * (output[:, :-1, :-1, :] - output[:, 1:, :-1, :])) / 2.0
    return tv_loss * weight

def targeted_attack_loss(pred, orig_pred, target, weight):
    """
    Note:
        balance can be adjusted by user for better results, range in [2,5] is recommended
    Arguments:
        pred {logits} -- Logits output by threat model (input: adv) 
        orig_pred {int} --  Original (correct) prediction by threat model
        target {int} -- Target lable assigned by user, range in [0,999]
        weight {float32} -- Attack weight assigned by user (args.attack_weight)
    
    Returns:
        [type] -- [description]
    """
    balance = 5
    orig_pred = np.eye(1000)[orig_pred]
    target = np.eye(1000)[target]
    loss1 = -1 * tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=orig_pred, logits=pred)
    loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=target, logits=pred)
    loss_attack = tf.reduce_sum(balance*loss2 + loss1) * weight
    return loss_attack


def untargeted_attack_loss(pred, orig_pred, weight):
    """
    Arguments:
        pred {logits} -- Logits output by threat model (input: adv)
        orig_pred {int} -- Original (correct) prediction by threat model
        weight {float32} -- Attack weight assigned by user (args.attack_weight)
    
    Returns:
        untargeted_attack_loss
    """
    orig_pred = np.eye(1000)[orig_pred]
    loss1 = -tf.nn.softmax_cross_entropy_with_logits_v2(labels=orig_pred, logits=pred)
    loss_attack = loss1
    return loss_attack * weight


def attack(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # prepare input images
    content_image,style_image,content_seg,style_seg = load_imgs_from_path(args)
    content_masks, style_masks = load_seg(content_seg, style_seg)
    
    input_image = tf.Variable(content_image)

    
    with tf.name_scope("constant"):
        vgg_const = Vgg19()
        resized_content_image = tf.image.resize_images(tf.constant(content_image), (224,224))
        vgg_const.fprop(resized_content_image)
        prob = sess.run(vgg_const.prob)
        pred = print_prob(prob[0], './synset.txt')
        args.true_label = np.argmax(prob)


        vgg_const.fprop(tf.constant(content_image), include_top=False)
        style_layers_const_c = [vgg_const.conv1_1, vgg_const.conv2_1, vgg_const.conv3_1, vgg_const.conv4_1,
                                vgg_const.conv5_1]
        content_fv, style_fvs_c = sess.run([vgg_const.conv4_2, style_layers_const_c])
        content_layer_const = tf.constant(content_fv)
        style_layers_const_c = [tf.constant(fv) for fv in style_fvs_c]

        vgg_const.fprop(tf.constant(style_image), include_top=False)
        style_layers_const = [vgg_const.conv1_1, vgg_const.conv2_1, vgg_const.conv3_1, vgg_const.conv4_1,
                              vgg_const.conv5_1]
        style_fvs = sess.run(style_layers_const)
        style_layers_const = [tf.constant(fv) for fv in style_fvs]
        del vgg_const 


    with tf.name_scope("variable"):
        vgg_var = Vgg19()
        vgg_var.fprop(input_image, include_top=False)
        style_layers_var = [vgg_var.conv1_1, vgg_var.conv2_1, vgg_var.conv3_1, vgg_var.conv4_1, vgg_var.conv5_1]
        content_layer_var = vgg_var.conv4_2
        layer_structure_all = [layer.name for layer in vgg_var.get_all_layers()]


    # Option loss: Content Loss
    loss_content = content_loss(content_layer_const, content_layer_var, float(args.content_weight))

    # Style Loss
    loss_styles_list = style_loss(layer_structure_all, style_layers_const, style_layers_const_c, style_layers_var, content_masks, style_masks, float(args.style_weight))
    
    loss_style = 0.0
    for loss in loss_styles_list:
        loss_style += loss

    with tf.name_scope("attack"):
        vgg_attack = Vgg19()
        content_width, content_height = content_image.shape[1], content_image.shape[0]
        physical_adaptor = Physical_Adaptor(args, content_seg,content_image, input_image,content_width, content_height)
        vgg_attack.fprop(physical_adaptor.resized_img)
        pred = vgg_attack.logits
        if args.targeted_attack == 1:
            loss_attack = targeted_attack_loss(pred=pred, orig_pred=args.true_label, target=args.target_label,
                                        weight=args.attack_weight)
        else:
            loss_attack = untargeted_attack_loss(pred=pred, orig_pred=args.true_label, weight=args.attack_weight)


    output_image = tf.squeeze(physical_adaptor.transformed_image , [0])

    loss_tv = total_variation_loss(input_image, float(args.tv_weight))
    
    
    total_loss = loss_tv + loss_content + loss_style + loss_attack
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
    grads = optimizer.compute_gradients(total_loss, [input_image])
    train_op = optimizer.apply_gradients(grads)

    sess.run(tf.global_variables_initializer())
    for i in range(0, args.max_iter+1):
        _, loss_content_, loss_styles_list_, loss_tv_, loss_attack_,overall_loss_, output_image_, prob= sess.run([
            train_op, loss_content, loss_styles_list, loss_tv, loss_attack, total_loss, output_image, vgg_attack.prob
        ],feed_dict={physical_adaptor.background: physical_adaptor.select_random_background(content_width,content_height)})
        pred, prob = print_prob(prob[0], './synset.txt')
        print('Iteration {} / {}\n\tContent loss: {}'.format(i, args.max_iter, loss_content_))
        for j, style_loss_ in enumerate(loss_styles_list_):
            print('\tStyle {} loss: {}'.format(j + 1, style_loss_))
        print('\tTV loss: {}'.format(loss_tv_))
        print('\tAttack loss: {}'.format(loss_attack_))
        print('\tTotal loss: {}'.format(overall_loss_ - loss_tv_))
        print('\tCurrent prediction: {}'.format(pred))

        if i % args.save_iter == 0:
            content_image_name = args.content_image_path.split('/')[-1].split('.')[0]
            suc = 'non'
            if args.targeted_attack == 1:
                if pred == args.target_label:
                    suc = 'suc'
        
            else:
                if pred != args.true_label:
                    suc = 'suc'
            save_result(output_image_, os.path.join(args.serial, suc +'_{}.jpg'.format(i)))

    sess.close()

