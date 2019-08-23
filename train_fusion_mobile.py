#!/usr/bin/env python3
from argparse import ArgumentParser
from datetime import timedelta
from importlib import import_module
import logging.config
import os
from signal import SIGINT, SIGTERM
import sys
import time

import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import common
import lbtoolbox as lb
import loss
from nets import NET_CHOICES
from heads import HEAD_CHOICES

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 动态申请显存，需要多少就申请多少

parser = ArgumentParser(description='Train a ReID network.')

# Required.
parser.add_argument('--validation_frequency', default = 100, type=common.positive_int,
                    help='validation steps') ######

parser.add_argument(
    '--validation_set',default='None', help='Path to the validation_set csv file.') ######

parser.add_argument(
    '--validation_image_root', type=common.readable_directory,
    help='Path that will be pre-pended to the filenames in the test_set csv.')   #######

parser.add_argument(
    '--experiment_root', required=True, type=common.writeable_directory,
    help='Location used to store checkpoints and dumped data.')

parser.add_argument(
    '--train_set',
    help='Path to the train_set csv file.')

parser.add_argument(
    '--image_root', type=common.readable_directory,
    help='Path that will be pre-pended to the filenames in the train_set csv.')

# Optional with sane defaults.

parser.add_argument(
    '--resume', action='store_true', default=False,
    help='When this flag is provided, all other arguments apart from the '
         'experiment_root are ignored and a previously saved set of arguments '
         'is loaded.')

##checkpoint number

parser.add_argument(
    '--checkpoint', default=None,
    help='Name of checkpoint file of the trained network within the experiment '
         'root. Uses the last checkpoint if not provided.')

parser.add_argument(
    '--model_name', default='mobilenet_v1_1_224', choices=NET_CHOICES,
    help='Name of the model to use.')

parser.add_argument(
    '--head_name', default='fusion_mobilenet', choices=HEAD_CHOICES,
    help='Name of the head to use.')

parser.add_argument(
    '--embedding_dim', default=128, type=common.positive_int,
    help='Dimensionality of the embedding space.')

parser.add_argument(
    '--initial_checkpoint', default=None,
    help='Path to the checkpoint file of the pretrained network.')

# TODO move these defaults to the .sh script?
parser.add_argument(
    '--batch_p', default=32, type=common.positive_int,
    help='The number P used in the PK-batches')

parser.add_argument(
    '--batch_k', default=4, type=common.positive_int,
    help='The numberK used in the PK-batches')

parser.add_argument(
    '--net_input_height', default=224, type=common.positive_int,
    help='Height of the input directly fed into the network.')

parser.add_argument(
    '--net_input_width', default=224, type=common.positive_int,
    help='Width of the input directly fed into the network.')

parser.add_argument(
    '--pre_crop_height', default=224, type=common.positive_int,
    help='Height used to resize a loaded image. This is ignored when no crop '
         'augmentation is applied.')

parser.add_argument(
    '--pre_crop_width', default=224, type=common.positive_int,
    help='Width used to resize a loaded image. This is ignored when no crop '
         'augmentation is applied.')
# TODO end

parser.add_argument(
    '--loading_threads', default=8, type=common.positive_int,
    help='Number of threads used for parallel loading.')

parser.add_argument(
    '--margin', default='soft', type=common.float_or_string,
    help='What margin to use: a float value for hard-margin, "soft" for '
         'soft-margin, or no margin if "none".')

parser.add_argument(
    '--metric', default='euclidean', choices=loss.cdist.supported_metrics,
    help='Which metric to use for the distance between embeddings.')

parser.add_argument(
    '--loss', default='batch_hard', choices=loss.LOSS_CHOICES.keys(),
    help='Enable the super-mega-advanced top-secret sampling stabilizer.')

parser.add_argument(
    '--learning_rate', default=3e-4, type=common.positive_float,
    help='The initial value of the learning-rate, before it kicks in.')

parser.add_argument(
    '--lr_decay_factor', default=0.96, type=common.positive_float,
    help='Learning rate decay factor')

parser.add_argument(
    '--lr_decay_steps', default=4000, type=common.positive_int,
    help='Learning rate decay steps')

parser.add_argument(
    '--train_iterations', default=25000, type=common.positive_int,
    help='Number of training iterations.')

parser.add_argument(
    '--decay_start_iteration', default=15000, type=int,
    help='At which iteration the learning-rate decay should kick-in.'
         'Set to -1 to disable decay completely.')

parser.add_argument(
    '--weight_decay_factor', default=0.001, type=common.positive_float,
    help='Weight decay factor')

parser.add_argument(
    '--checkpoint_frequency', default=1000, type=common.nonnegative_int,
    help='After how many iterations a checkpoint is stored. Set this to 0 to '
         'disable intermediate storing. This will result in only one final '
         'checkpoint.')

parser.add_argument(
    '--flip_augment', action='store_true', default=False,
    help='When this flag is provided, flip augmentation is performed.')

parser.add_argument(
    '--crop_augment', action='store_true', default=False,
    help='When this flag is provided, crop augmentation is performed. Based on'
         'The `crop_height` and `crop_width` parameters. Changing this flag '
         'thus likely changes the network input size!')

parser.add_argument(
    '--detailed_logs', action='store_true', default=False,
    help='Store very detailed logs of the training in addition to TensorBoard'
         ' summaries. These are mem-mapped numpy files containing the'
         ' embeddings, losses and FIDs seen in each batch during training.'
         ' Everything can be re-constructed and analyzed that way.')



def sample_k_fids_for_pid(pid, all_fids, all_pids, batch_k):
    """ Given a PID, select K FIDs of that specific PID. """
    possible_fids = tf.boolean_mask(all_fids, tf.equal(all_pids, pid))

    # The following simply uses a subset of K of the possible FIDs
    # if more than, or exactly K are available. Otherwise, we first
    # create a padded list of indices which contain a multiple of the
    # original FID count such that all of them will be sampled equally likely.
    count = tf.shape(possible_fids)[0]
    padded_count = tf.cast(tf.ceil(batch_k / count), tf.int32) * count
    full_range = tf.mod(tf.range(padded_count), count)

    # Sampling is always performed by shuffling and taking the first k.
    shuffled = tf.random_shuffle(full_range)
    selected_fids = tf.gather(possible_fids, shuffled[:batch_k])

    return selected_fids, tf.fill([batch_k], pid)


def main():
    args = parser.parse_args()

    # We store all arguments in a json file. This has two advantages:
    # 1. We can always get back and see what exactly that experiment was
    # 2. We can resume an experiment as-is without needing to remember all flags.
    args_file = os.path.join(args.experiment_root, 'args.json')
    if args.resume:
        if not os.path.isfile(args_file):
            raise IOError('`args.json` not found in {}'.format(args_file))

        print('Loading args from {}.'.format(args_file))
        with open(args_file, 'r') as f:
            args_resumed = json.load(f)
        args_resumed['resume'] = True  # This would be overwritten.

        # When resuming, we not only want to populate the args object with the
        # values from the file, but we also want to check for some possible
        # conflicts between loaded and given arguments.
        for key, value in args.__dict__.items():
            if key in args_resumed:
                resumed_value = args_resumed[key]
                if resumed_value != value:
                    print('Warning: For the argument `{}` we are using the'
                          ' loaded value `{}`. The provided value was `{}`'
                          '.'.format(key, resumed_value, value))
                    comand = input('Would you like to restore it?(yes/no)')
                    if comand == 'yes':
                        args.__dict__[key] = resumed_value
                        print('For the argument `{}` we are using the loaded value `{}`.'.format(key,
                                                                                                 args.__dict__[key]))
                    else:
                        print('For the argument `{}` we are using the provided value `{}`.'.format(key,
                                                                                                   args.__dict__[key]))
            else:
                print('Warning: A new argument was added since the last run:'
                      ' `{}`. Using the new value: `{}`.'.format(key, value))
        os.remove(args_file)
        with open(args_file, 'w') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)

    else:
        # If the experiment directory exists already, we bail in fear.
        if os.path.exists(args.experiment_root):
            if os.listdir(args.experiment_root):
                print('The directory {} already exists and is not empty.'
                      ' If you want to resume training, append --resume to'
                      ' your call.'.format(args.experiment_root))
                exit(1)
        else:
            os.makedirs(args.experiment_root)

        # Store the passed arguments for later resuming and grepping in a nice
        # and readable format.
        with open(args_file, 'w') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)

    log_file = os.path.join(args.experiment_root, "train")
    logging.config.dictConfig(common.get_logging_dict(log_file))
    log = logging.getLogger('train')

    # Also show all parameter values at the start, for ease of reading logs.
    log.info('Training using the following parameters:')
    for key, value in sorted(vars(args).items()):
        log.info('{}: {}'.format(key, value))

    # Check them here, so they are not required when --resume-ing.
    if not args.train_set:
        parser.print_help()
        log.error("You did not specify the `train_set` argument!")
        sys.exit(1)
    if not args.image_root:
        parser.print_help()
        log.error("You did not specify the required `image_root` argument!")
        sys.exit(1)
######################################################################################
    # prepare the training dataset
    # Load the data from the csv file. see Common.load_dataset function for details
    pids, fids = common.load_dataset(args.train_set, args.image_root)
    max_fid_len = max(map(len, fids))  # We'll need this later for logfiles.

    # Setup a tf.Dataset where one "epoch" loops over all PIDS.
    # PIDS are shuffled after every epoch and continue indefinitely.
    unique_pids = np.unique(pids)
    dataset = tf.data.Dataset.from_tensor_slices(unique_pids)
    dataset = dataset.shuffle(len(unique_pids))

    # Constrain the dataset size to a multiple of the batch-size, so that
    # we don't get overlap at the end of each epoch.
    dataset = dataset.take((len(unique_pids) // args.batch_p) * args.batch_p)
    dataset = dataset.repeat(None)  # Repeat forever. Funny way of stating it.

    # For every PID, get K images.
    dataset = dataset.map(lambda pid: sample_k_fids_for_pid(
        pid, all_fids=fids, all_pids=pids, batch_k=args.batch_k))  # now the dataset has been modified as [selected_fids
    # , pid] due to the return of the function 'sample_k_fids_for_pid'

    # Ungroup/flatten the batches for easy loading of the files.
    dataset = dataset.apply(tf.contrib.data.unbatch())

    # Convert filenames to actual image tensors.
    net_input_size = (args.net_input_height, args.net_input_width)
    pre_crop_size = (args.pre_crop_height, args.pre_crop_width)
    dataset = dataset.map(
        lambda fid, pid: common.fid_to_image(
            fid, pid, image_root=args.image_root,
            image_size=pre_crop_size if args.crop_augment else net_input_size),
        num_parallel_calls=args.loading_threads)  # now the dataset has been modified as [selected_images
    # , fid， pid] due to the return of the function 'fid_to_image'


    # Augment the data if specified by the arguments.
    if args.flip_augment:
        dataset = dataset.map(
            lambda im, fid, pid: (tf.image.random_flip_left_right(im), fid, pid))
    if args.crop_augment:
        dataset = dataset.map(
            lambda im, fid, pid: (tf.random_crop(im, net_input_size + (3,)), fid, pid))

    # Group it back into PK batches.
    batch_size = args.batch_p * args.batch_k
    dataset = dataset.batch(batch_size)

    # Overlap producing and consuming for parallelism.
    dataset = dataset.prefetch(1)

    # Since we repeat the data infinitely, we only need a one-shot iterator.
    images_train, fids_train, pids_train = dataset.make_one_shot_iterator().get_next()

#######################################################################################################################
    # prepare the validation set
    pids_val, fids_val = common.load_dataset(args.validation_set, args.validation_image_root)
    # Setup a tf.Dataset where one "epoch" loops over all PIDS.
    # PIDS are shuffled after every epoch and continue indefinitely.
    unique_pids_val = np.unique(pids_val)
    dataset_val = tf.data.Dataset.from_tensor_slices(unique_pids_val)
    dataset_val = dataset_val.shuffle(len(unique_pids_val))

    # Constrain the dataset size to a multiple of the batch-size, so that
    # we don't get overlap at the end of each epoch.
    dataset_val = dataset_val.take((len(unique_pids_val) // args.batch_p) * args.batch_p)
    dataset_val = dataset_val.repeat(None)  # Repeat forever. Funny way of stating it.

    # For every PID, get K images.
    dataset_val = dataset_val.map(lambda pid: sample_k_fids_for_pid(
        pid, all_fids=fids_val, all_pids=pids_val,
        batch_k=args.batch_k))  # now the dataset has been modified as [selected_fids
    # , pid] due to the return of the function 'sample_k_fids_for_pid'

    # Ungroup/flatten the batches for easy loading of the files.
    dataset_val = dataset_val.apply(tf.contrib.data.unbatch())

    # Convert filenames to actual image tensors.
    net_input_size = (args.net_input_height, args.net_input_width)
    pre_crop_size = (args.pre_crop_height, args.pre_crop_width)
    dataset_val = dataset_val.map(
        lambda fid, pid: common.fid_to_image(
            fid, pid, image_root=args.validation_image_root,
            image_size=pre_crop_size if args.crop_augment else net_input_size),
        num_parallel_calls=args.loading_threads)  # now the dataset has been modified as [selected_images
    # , fid， pid] due to the return of the function 'fid_to_image'

    # Augment the data if specified by the arguments.
    if args.flip_augment:
        dataset_val = dataset_val.map(
            lambda im, fid, pid: (tf.image.random_flip_left_right(im), fid, pid))
    if args.crop_augment:
        dataset_val = dataset_val.map(
            lambda im, fid, pid: (tf.random_crop(im, net_input_size + (3,)), fid, pid))

    # Group it back into PK batches.
    dataset_val = dataset_val.batch(batch_size)

    # Overlap producing and consuming for parallelism.
    dataset_val = dataset_val.prefetch(1)

    # Since we repeat the data infinitely, we only need a one-shot iterator.
    images_val, fids_val, pids_val = dataset_val.make_one_shot_iterator().get_next()
########################################################################################################################

    # Create the model and an embedding head.
    model = import_module('nets.' + args.model_name)
    head = import_module('heads.' + args.head_name)

    # Feed the image through the model. The returned `body_prefix` will be used
    # further down to load the pre-trained weights for all variables with this
    # prefix.
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, args.net_input_height, args.net_input_width, 3],
                                  name='input')
    pids = tf.placeholder(dtype=tf.string, shape=[None, ], name='pids')
    fids = tf.placeholder(dtype=tf.string, shape=[None, ], name='fids')

    endpoints, body_prefix = model.endpoints(input_images, is_training=True)
    with tf.name_scope('head'):
        endpoints = head.head(endpoints, args.embedding_dim, is_training=True)

    # Create the loss in two steps:
    # 1. Compute all pairwise distances according to the specified metric.
    # 2. For each anchor along the first dimension, compute its loss.
    # dists = loss.cdist(endpoints['emb'], endpoints['emb'], metric=args.metric)
    # losses, train_top1, prec_at_k, _, neg_dists, pos_dists = loss.LOSS_CHOICES[args.loss](
    #     dists, pids, args.margin, batch_precision_at_k=args.batch_k-1)
    # # '_' stands for the boolean matrix shows topK where the correct match of the identities occurs
    # shape=(batch_size,K)

# 更改
    # loss1
    dists1 = loss.cdist(endpoints['feature1'],endpoints['feature1'],metric=args.metric)
    losses1,_,_,_,_,_ =loss.LOSS_CHOICES[args.loss](
        dists1, pids, args.margin, batch_precision_at_k=args.batch_k - 1)
    dists2 = loss.cdist(endpoints['feature2'], endpoints['feature2'], metric=args.metric)
    losses2, _, _, _, _, _ = loss.LOSS_CHOICES[args.loss](
        dists2, pids, args.margin, batch_precision_at_k=args.batch_k - 1)
    dists3 = loss.cdist(endpoints['feature3'], endpoints['feature3'], metric=args.metric)
    losses3, _, _, _, _, _ = loss.LOSS_CHOICES[args.loss](
        dists3, pids, args.margin, batch_precision_at_k=args.batch_k - 1)
    dists4 = loss.cdist(endpoints['feature4'], endpoints['feature4'], metric=args.metric)
    losses4, _, _, _, _, _ = loss.LOSS_CHOICES[args.loss](
        dists4, pids, args.margin, batch_precision_at_k=args.batch_k - 1)
    dists5 = loss.cdist(endpoints['feature5'], endpoints['feature5'], metric=args.metric)
    losses5, _, _, _, _, _ = loss.LOSS_CHOICES[args.loss](
        dists5, pids, args.margin, batch_precision_at_k=args.batch_k - 1)
    dists_fu = loss.cdist(endpoints['fusion_layer'], endpoints['fusion_layer'], metric=args.metric)
    losses_fu, train_top1, prec_at_k, _, neg_dists, pos_dists = loss.LOSS_CHOICES[args.loss](
        dists_fu, pids, args.margin, batch_precision_at_k=args.batch_k - 1)

    losses = losses1+losses2+losses3+losses4+losses5+losses_fu

    # Count the number of active entries, and compute the total batch loss.
    num_active = tf.reduce_sum(tf.cast(tf.greater(losses, 1e-5), tf.float32))

    # 此处losses即为 pospair 比 negpair+margin 还大的部分
    loss_mean = tf.reduce_mean(losses)

    # Some logging for tensorboard.
    tf.summary.histogram('loss_distribution', losses)
    tf.summary.scalar('loss', loss_mean)
    tf.summary.scalar('batch_top1', train_top1)
    tf.summary.scalar('batch_prec_at_{}'.format(args.batch_k-1), prec_at_k)
    tf.summary.scalar('active_count', num_active)
    #tf.summary.histogram('embedding_dists', dists)
    tf.summary.histogram('embedding_pos_dists', pos_dists)
    tf.summary.histogram('embedding_neg_dists', neg_dists)
    tf.summary.histogram('embedding_lengths',
                         tf.norm(endpoints['emb_raw'], axis=1))

    # Create the mem-mapped arrays in which we'll log all training detail in
    # addition to tensorboard, because tensorboard is annoying for detailed
    # inspection and actually discards data in histogram summaries.
    if args.detailed_logs:
        log_embs = lb.create_or_resize_dat(
            os.path.join(args.experiment_root, 'embeddings'),
            dtype=np.float32, shape=(args.train_iterations, batch_size, args.embedding_dim))
        log_loss = lb.create_or_resize_dat(
            os.path.join(args.experiment_root, 'losses'),
            dtype=np.float32, shape=(args.train_iterations, batch_size))
        log_fids = lb.create_or_resize_dat(
            os.path.join(args.experiment_root, 'fids'),
            dtype='S' + str(max_fid_len), shape=(args.train_iterations, batch_size))

    # These are collected here before we add the optimizer, because depending
    # on the optimizer, it might add extra slots, which are also global
    # variables, with the exact same prefix.
    model_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, body_prefix)

    # Define the optimizer and the learning-rate schedule.
    # Unfortunately, we get NaNs if we don't handle no-decay separately.
    global_step = tf.Variable(0, name='global_step', trainable=False)  # 'global_step' means the number of batches seen
                                                                       #  by graph
    if 0 <= args.decay_start_iteration < args.train_iterations:
        learning_rate = tf.train.exponential_decay(
            args.learning_rate,
            tf.maximum(0, global_step - args.decay_start_iteration),  # decay every 'lr_decay_steps' after the
                                                                      # 'decay_start_iteration'
            # args.train_iterations - args.decay_start_iteration, args.weight_decay_factor)
            args.lr_decay_steps, args.lr_decay_factor, staircase=True)
    else:
        learning_rate = args.learning_rate  # the case when we set 'decay_start_iteration' as -1
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate,epsilon=1e-3)
    # Feel free to try others!
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate)

    # Update_ops are used to update batchnorm stats.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss_mean, global_step=global_step)

    # Define a saver for the complete model.
    checkpoint_saver = tf.train.Saver(max_to_keep=0)

    with tf.Session(config=config) as sess:
        if args.resume:
            # In case we're resuming, simply load the full checkpoint to init.
            if args.checkpoint is None:
                last_checkpoint = tf.train.latest_checkpoint(args.experiment_root)
                log.info('Restoring from checkpoint: {}'.format(last_checkpoint))
                checkpoint_saver.restore(sess, last_checkpoint)
            else:
                ckpt_path = os.path.join(args.experiment_root, args.checkpoint)
                log.info('Restoring from checkpoint: {}'.format(args.checkpoint))
                checkpoint_saver.restore(sess, ckpt_path)
        else:
            # But if we're starting from scratch, we may need to load some
            # variables from the pre-trained weights, and random init others.
            sess.run(tf.global_variables_initializer())
            if args.initial_checkpoint is not None:
                saver = tf.train.Saver(model_variables)
                saver.restore(sess, args.initial_checkpoint)  # restore the pre-trained parameter from online model

            # In any case, we also store this initialization as a checkpoint,
            # such that we could run exactly reproduceable experiments.
            checkpoint_saver.save(sess, os.path.join(
                args.experiment_root, 'checkpoint'), global_step=0)

        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.experiment_root, sess.graph)

        start_step = sess.run(global_step)
        log.info('Starting training from iteration {}.'.format(start_step))

        # Finally, here comes the main-loop. This `Uninterrupt` is a handy
        # utility such that an iteration still finishes on Ctrl+C and we can
        # stop the training cleanly.
        with lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:
            for i in range(start_step, args.train_iterations):

                # Compute gradients, update weights, store logs!
                start_time = time.time()
                # _, summary, step, b_prec_at_k, b_embs, b_loss, b_fids = \
                #     sess.run([train_op, merged_summary, global_step,
                #               prec_at_k, endpoints['emb'], losses, fids])
                _, summary, step, b_prec_at_k, b_embs, b_loss, b_fids = \
                    sess.run([train_op, merged_summary, global_step,
                              prec_at_k, endpoints['emb'], losses, fids], feed_dict={input_images: images_train.eval(),
                                                                                     pids: pids_train.eval(),
                                                                                     fids: fids_train.eval()})
                elapsed_time = time.time() - start_time

                # Compute the iteration speed and add it to the summary.
                # We did observe some weird spikes that we couldn't track down.
                summary2 = tf.Summary()
                summary2.value.add(tag='secs_per_iter', simple_value=elapsed_time)
                summary_writer.add_summary(summary2, step)
                summary_writer.add_summary(summary, step)

                if args.detailed_logs:
                    log_embs[i], log_loss[i], log_fids[i] = b_embs, b_loss, b_fids

                # Do a huge print out of the current progress.
                seconds_todo = (args.train_iterations - step) * elapsed_time
                log.info('iter:{:6d}, loss min|avg|max: {:.3f}|{:.3f}|{:6.3f}, '
                         'batch-p@{}: {:.2%}, ETA: {} ({:.2f}s/it)'.format(
                             step,
                             float(np.min(b_loss)),
                             float(np.mean(b_loss)),
                             float(np.max(b_loss)),
                             args.batch_k-1, float(b_prec_at_k),
                             timedelta(seconds=int(seconds_todo)),
                             elapsed_time))
                sys.stdout.flush()
                sys.stderr.flush()

                # Save a checkpoint of training every so often.
                if (args.checkpoint_frequency > 0 and
                        step % args.checkpoint_frequency == 0):
                    checkpoint_saver.save(sess, os.path.join(
                        args.experiment_root, 'checkpoint'), global_step=step)

                # get validation results
                if (args.validation_frequency > 0 and
                        step % args.validation_frequency == 0):
                    b_prec_at_k_val, b_loss, b_fids = \
                        sess.run([prec_at_k, losses, fids], feed_dict={input_images: images_val.eval(),
                                                                       pids: pids_val.eval(),
                                                                       fids: fids_val.eval()})
                    log.info('Validation @:{:6d} iteration, loss min|avg|max: {:.3f}|{:.3f}|{:6.3f}, '
                             'batch-p@{}: {:.2%}'.format(
                        step,
                        float(np.min(b_loss)),
                        float(np.mean(b_loss)),
                        float(np.max(b_loss)),
                        args.batch_k - 1, float(b_prec_at_k_val)
                    ))
                    sys.stdout.flush()
                    sys.stderr.flush()
                    summary3 = tf.Summary()
                    summary3.value.add(tag='validationloss', simple_value=float(np.mean(b_loss)))
                    summary_writer.add_summary(summary3, step)
                    summary_writer.add_summary(summary3, step)

                # Stop the main-loop at the end of the step, if requested.
                if u.interrupted:
                    log.info("Interrupted on request!")
                    break

        # Store one final checkpoint. This might be redundant, but it is crucial
        # in case intermediate storing was disabled and it saves a checkpoint
        # when the process was interrupted.
        checkpoint_saver.save(sess, os.path.join(
            args.experiment_root, 'checkpoint'), global_step=step)


if __name__ == '__main__':
    main()

