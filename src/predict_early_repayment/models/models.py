import logging
from datetime import datetime

import pandas as pd
import tensorflow as tf
from predict_early_repayment.models.tabnet import *
from predict_early_repayment.utils import load_from_json

logger = logging.getLogger(__name__)


def define_tabnet_model(
    train_batch, test_batch, val_batch, tabnet_model, model_save_path
):
    """General function to prepare tabnet model."""

    # load config
    tabnet_config = load_from_json("commons/tabnet_config.json")

    # prepare data
    train_iter = train_batch.make_initializable_iterator()
    test_iter = test_batch.make_initializable_iterator()
    val_iter = val_batch.make_initializable_iterator()

    feature_train_batch, label_train_batch = train_iter.get_next()
    feature_test_batch, label_test_batch = test_iter.get_next()
    feature_val_batch, label_val_batch = val_iter.get_next()

    # Define the model and losses
    encoded_train_batch, total_entropy = tabnet_model.encoder(
        feature_train_batch, reuse=False, is_training=True
    )

    logits_orig_batch, _ = tabnet_model.classify(encoded_train_batch, reuse=False)

    softmax_orig_key_op = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_orig_batch, labels=label_train_batch
        )
    )

    train_loss_op = (
        softmax_orig_key_op + tabnet_config["SPARSITY_LOSS_WEIGHT"] * total_entropy
    )

    tf.summary.scalar("Total loss", train_loss_op)

    # Optimization step
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        tabnet_config["INIT_LEARNING_RATE"],
        global_step=global_step,
        decay_steps=tabnet_config["DECAY_EVERY"],
        decay_rate=tabnet_config["DECAY_RATE"],
    )
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        gvs = optimizer.compute_gradients(train_loss_op)
        capped_gvs = [
            (
                tf.clip_by_value(
                    grad,
                    -tabnet_config["GRADIENT_THRESH"],
                    tabnet_config["GRADIENT_THRESH"],
                ),
                var,
            )
            for grad, var in gvs
        ]
        train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

    # Model evaluation

    # Test performance
    encoded_test_batch, _ = tabnet_model.encoder(
        feature_test_batch, reuse=True, is_training=False
    )
    _, prediction_test = tabnet_model.classify(encoded_test_batch, reuse=True)
    predicted_labels = tf.cast(tf.argmax(prediction_test, 1), dtype=tf.int32)
    test_eq_op = tf.equal(predicted_labels, label_test_batch)
    test_acc_op = tf.reduce_mean(tf.cast(test_eq_op, dtype=tf.float32))

    # Validation performance
    encoded_val_batch, _ = tabnet_model.encoder(
        feature_val_batch, reuse=True, is_training=False
    )
    _, prediction_val = tabnet_model.classify(encoded_val_batch, reuse=True)
    # validation_pred_labels = tf.cast(tf.argmax(prediction_val, 1), dtype=tf.int32)

    tf.summary.scalar("Test_accuracy", test_acc_op)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = tabnet_config["MODEL_NAME"] + f"_{current_time}"
    init = tf.initialize_all_variables()
    init_local = tf.local_variables_initializer()
    init_table = tf.tables_initializer(name="Initialize_all_tables")
    saver = tf.train.Saver()
    summaries = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter("./tflog/" + model_name, sess.graph)

        sess.run(init)
        sess.run(init_local)
        sess.run(init_table)
        sess.run(train_iter.initializer)
        sess.run(test_iter.initializer)
        sess.run(val_iter.initializer)

        for step in range(1, tabnet_config["MAX_STEPS"] + 1):
            if step % tabnet_config["DISPLAY_STEP"] == 0:
                _, train_loss, merged_summary = sess.run(
                    [train_op, train_loss_op, summaries]
                )
                summary_writer.add_summary(merged_summary, step)
                logger.info(
                    "Step "
                    + str(step)
                    + ", Training Loss = "
                    + "{:.4f}".format(train_loss)
                )
            else:
                _ = sess.run(train_op)

            if step % tabnet_config["TEST_STEP"] == 0:
                feed_arr = [vars()["summaries"], vars()["test_acc_op"]]

                test_arr = sess.run(feed_arr)
                merged_summary = test_arr[0]
                test_acc = test_arr[1]

                logger.info(
                    "Step "
                    + str(step)
                    + ", Test Accuracy = "
                    + "{:.4f}".format(test_acc)
                )
                summary_writer.add_summary(merged_summary, step)

            if step % tabnet_config["SAVE_STEP"] == 0:
                saver.save(sess, "./checkpoints/" + model_name + ".ckpt")

        output = sess.run([prediction_val])
        test_labels = sess.run([predicted_labels])
        test_perc = sess.run([prediction_test])
        pd.DataFrame(test_perc[0]).to_csv(
            f"{model_save_path}/test_output.csv", index=False
        )
        pd.DataFrame(test_labels[0]).to_csv(
            f"{model_save_path}/test_labels.csv", index=False
        )
        pd.DataFrame(output[0]).to_csv(f"{model_save_path}/val_output.csv", index=False)
        saver.save(sess, "./checkpoints/" + model_name + ".ckpt")

    return
