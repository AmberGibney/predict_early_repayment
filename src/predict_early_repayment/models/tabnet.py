import functools
import logging
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow.compat.v1 as tf
from absl import app
from tensorflow.python.framework import dtypes

logger = logging.getLogger(__name__)

sys.path.append("..")

tf.compat.v1.disable_eager_execution()

# -*- coding: utf-8 -*-


tf.reset_default_graph()


def set_defaults(
    int_columns, encoded_categorical_columns, bool_columns, float_columns, str_columns
):
    return (
        [[0] for col in int_columns]
        + [[0] for col in encoded_categorical_columns]
        + [[0] for col in bool_columns]
        + [[0.0] for col in float_columns]
        + [[""] for col in str_columns]
        + [[-1]]
    )


def get_columns(
    int_columns, encoded_categorical_columns, bool_columns, float_columns, str_columns
):
    """Get the representations for all input columns."""

    columns = []
    if float_columns:
        columns += [
            tf.feature_column.numeric_column(ci, dtype=dtypes.float32)
            for ci in float_columns
        ]
    if int_columns:
        columns += [
            tf.feature_column.numeric_column(ci, dtype=dtypes.int32)
            for ci in int_columns
        ]
    if encoded_categorical_columns:
        columns += [
            tf.feature_column.numeric_column(ci, dtype=dtypes.int32)
            for ci in encoded_categorical_columns
        ]
    if str_columns:
        # pylint: disable=g-complex-comprehension
        str_nuniquess = len(set(str_columns))
        columns += [
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_hash_bucket(
                    ci, hash_bucket_size=int(3 * num)
                ),
                dimension=1,
            )
            for ci, num in zip(str_columns, str_nuniquess)
        ]
    if bool_columns:
        # pylint: disable=g-complex-comprehension
        columns += [
            tf.feature_column.numeric_column(ci, dtype=dtypes.int32)
            for ci in bool_columns
        ]
    return columns


def parse_csv(
    int_columns,
    encoded_categorical_columns,
    bool_columns,
    float_columns,
    str_columns,
    label_column,
    value_column,
):
    """Parses a CSV file based on the provided column types."""
    defaults = set_defaults(
        int_columns,
        encoded_categorical_columns,
        bool_columns,
        float_columns,
        str_columns,
    )
    all_columns = (
        int_columns
        + encoded_categorical_columns
        + bool_columns
        + float_columns
        + str_columns
        + [label_column]
    )
    columns = tf.io.decode_csv(value_column, record_defaults=defaults)
    features = dict(zip(all_columns, columns))
    label = features.pop(label_column)
    classes = tf.cast(label, tf.int32)
    return features, classes


def input_fn(
    data_file,
    int_columns,
    encoded_categorical_columns,
    bool_columns,
    float_columns,
    str_columns,
    label_column,
    num_epochs,
    shuffle,
    batch_size,
    n_buffer=50,
    n_parallel=16,
):
    """Function to read the input file and return the dataset.

    Args:
        data_file: Name of the file.
        num_epochs: Number of epochs.
        shuffle: Whether to shuffle the data.
        batch_size: Batch size.
        n_buffer: Buffer size.
        n_parallel: Number of cores for multi-core processing option.

    Returns:
        The Tensorflow dataset.

    """

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file).skip(1)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=n_buffer)

    parse_csv_partial = functools.partial(
        parse_csv,
        int_columns,
        encoded_categorical_columns,
        bool_columns,
        float_columns,
        str_columns,
        label_column,
    )

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(parse_csv_partial, num_parallel_calls=n_parallel)

    # Repeat after shuffling, to prevent separate epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    return dataset


def sort_col_names(feature_cols):
    column_names = sorted(feature_cols)
    logger.info(
        "Ordered column names, corresponding to the indexing in Tensorboard visualization"
    )
    for fi in range(len(column_names)):
        logger.info(str(fi) + " : " + column_names[fi])


def glu(act, n_units):
    """Generalized linear unit nonlinear activation."""
    return act[:, :n_units] * tf.nn.sigmoid(act[:, n_units:])


class TabNet(object):
    """TabNet model class."""

    def __init__(
        self,
        columns,
        num_features,
        feature_dim,
        output_dim,
        num_decision_steps,
        relaxation_factor,
        batch_momentum,
        virtual_batch_size,
        num_classes,
        epsilon=0.00001,
    ):
        """Initializes a TabNet instance.

        Args:
          columns: The Tensorflow column names for the dataset.
          num_features: The number of input features (i.e the number of columns for
            tabular data assuming each feature is represented with 1 dimension).
          feature_dim: Dimensionality of the hidden representation in feature
            transformation block. Each layer first maps the representation to a
            2*feature_dim-dimensional output and half of it is used to determine the
            nonlinearity of the GLU activation where the other half is used as an
            input to GLU, and eventually feature_dim-dimensional output is
            transferred to the next layer.
          output_dim: Dimensionality of the outputs of each decision step, which is
            later mapped to the final classification or regression output.
          num_decision_steps: Number of sequential decision steps.
          relaxation_factor: Relaxation factor that promotes the reuse of each
            feature at different decision steps. When it is 1, a feature is enforced
            to be used only at one decision step and as it increases, more
            flexibility is provided to use a feature at multiple decision steps.
          batch_momentum: Momentum in ghost batch normalization.
          virtual_batch_size: Virtual batch size in ghost batch normalization. The
            overall batch size should be an integer multiple of virtual_batch_size.
          num_classes: Number of output classes.
          epsilon: A small number for numerical stability of the entropy calcations.

        Returns:
          A TabNet instance.

        """

        self.columns = columns
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_classes = num_classes
        self.epsilon = epsilon

    def encoder(self, data, reuse, is_training):
        """TabNet encoder model."""

        with tf.variable_scope("Encoder", reuse=reuse):
            # Reads and normalizes input features.
            features = tf.feature_column.input_layer(data, self.columns)
            features = tf.layers.batch_normalization(
                features, training=is_training, momentum=self.batch_momentum
            )
            batch_size = tf.shape(features)[0]

            # Initializes decision-step dependent variables.
            output_aggregated = tf.zeros([batch_size, self.output_dim])
            masked_features = features
            mask_values = tf.zeros([batch_size, self.num_features])
            aggregated_mask_values = tf.zeros([batch_size, self.num_features])
            complemantary_aggregated_mask_values = tf.ones(
                [batch_size, self.num_features]
            )
            total_entropy = 0

            v_b = self.virtual_batch_size if is_training else 1
            for ni in range(self.num_decision_steps):
                # Feature transformer with two shared and two decision step dependent
                # blocks is used below.
                reuse_flag = ni > 0

                transform_f1 = tf.layers.dense(
                    masked_features,
                    self.feature_dim * 2,
                    name="Transform_f1",
                    reuse=reuse_flag,
                    use_bias=False,
                )
                transform_f1 = tf.layers.batch_normalization(
                    transform_f1,
                    training=is_training,
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b,
                )
                transform_f1 = glu(transform_f1, self.feature_dim)

                transform_f2 = tf.layers.dense(
                    transform_f1,
                    self.feature_dim * 2,
                    name="Transform_f2",
                    reuse=reuse_flag,
                    use_bias=False,
                )
                transform_f2 = tf.layers.batch_normalization(
                    transform_f2,
                    training=is_training,
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b,
                )
                transform_f2 = (
                    glu(transform_f2, self.feature_dim) + transform_f1
                ) * np.sqrt(0.5)

                transform_f3 = tf.layers.dense(
                    transform_f2,
                    self.feature_dim * 2,
                    name="Transform_f3" + str(ni),
                    use_bias=False,
                )
                transform_f3 = tf.layers.batch_normalization(
                    transform_f3,
                    training=is_training,
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b,
                )
                transform_f3 = (
                    glu(transform_f3, self.feature_dim) + transform_f2
                ) * np.sqrt(0.5)

                transform_f4 = tf.layers.dense(
                    transform_f3,
                    self.feature_dim * 2,
                    name="Transform_f4" + str(ni),
                    use_bias=False,
                )
                transform_f4 = tf.layers.batch_normalization(
                    transform_f4,
                    training=is_training,
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b,
                )
                transform_f4 = (
                    glu(transform_f4, self.feature_dim) + transform_f3
                ) * np.sqrt(0.5)

                if ni > 0:
                    decision_out = tf.nn.relu(transform_f4[:, : self.output_dim])

                    # Decision aggregation.
                    output_aggregated += decision_out

                    # Aggregated masks are used for visualization of the
                    # feature importance attributes.
                    scale_agg = tf.reduce_sum(decision_out, axis=1, keep_dims=True) / (
                        self.num_decision_steps - 1
                    )
                    aggregated_mask_values += mask_values * scale_agg

                features_for_coef = transform_f4[:, self.output_dim :]

                if ni < self.num_decision_steps - 1:
                    # Determines the feature masks via linear and nonlinear
                    # transformations, taking into account of aggregated feature use.
                    mask_values = tf.layers.dense(
                        features_for_coef,
                        self.num_features,
                        name="Transform_coef" + str(ni),
                        use_bias=False,
                    )
                    mask_values = tf.layers.batch_normalization(
                        mask_values,
                        training=is_training,
                        momentum=self.batch_momentum,
                        virtual_batch_size=v_b,
                    )
                    mask_values *= complemantary_aggregated_mask_values
                    mask_values = tf.contrib.sparsemax.sparsemax(mask_values)

                    # Relaxation factor controls the amount of reuse of features between
                    # different decision blocks and updated with the values of
                    # coefficients.
                    complemantary_aggregated_mask_values *= (
                        self.relaxation_factor - mask_values
                    )

                    # Entropy is used to penalize the amount of sparsity in feature
                    # selection.
                    total_entropy += tf.reduce_mean(
                        tf.reduce_sum(
                            -mask_values * tf.log(mask_values + self.epsilon), axis=1
                        )
                    ) / (self.num_decision_steps - 1)

                    # Feature selection.
                    masked_features = tf.multiply(mask_values, features)

                    # Visualization of the feature selection mask at decision step ni
                    tf.summary.image(
                        "Mask_for_step" + str(ni),
                        tf.expand_dims(tf.expand_dims(mask_values, 0), 3),
                        max_outputs=1,
                    )

            # Visualization of the aggregated feature importances
            tf.summary.image(
                "Aggregated_mask",
                tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3),
                max_outputs=1,
            )

            return output_aggregated, total_entropy

    def classify(self, activations, reuse):
        """TabNet classify block."""

        with tf.variable_scope("Classify", reuse=reuse):
            logits = tf.layers.dense(activations, self.num_classes, use_bias=False)
            predictions = tf.nn.softmax(logits)
            return logits, predictions

    def regress(self, activations, reuse):
        """TabNet regress block."""

        with tf.variable_scope("Regress", reuse=reuse):
            predictions = tf.layers.dense(activations, 1)
            return predictions


# taken from https://gist.github.com/justheuristic/60167e77a95221586be315ae527c3cbd
def entmax15(inputs, axis=-1):
    """Entmax 1.5 implementation, heavily inspired by.

     * paper: https://arxiv.org/pdf/1905.05702.pdf
     * pytorch code: https://github.com/deep-spin/entmax
    :param inputs: similar to softmax logits, but for entmax1.5
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    :return: entmax activations of same shape as inputs

    """

    @tf.custom_gradient
    def _entmax_inner(inputs):
        with tf.name_scope("entmax"):
            inputs = inputs / 2  # divide by 2 so as to solve actual entmax
            # subtract max for stability
            inputs -= tf.reduce_max(inputs, axis, keep_dims=True)

            threshold, _ = entmax_threshold_and_support(inputs, axis)
            outputs_sqrt = tf.nn.relu(inputs - threshold)
            outputs = tf.square(outputs_sqrt)

        def grad_fn(d_outputs):
            with tf.name_scope("entmax_grad"):
                d_inputs = d_outputs * outputs_sqrt
                q = tf.reduce_sum(d_inputs, axis=axis, keep_dims=True)
                q = q / tf.reduce_sum(outputs_sqrt, axis=axis, keep_dims=True)
                d_inputs -= q * outputs_sqrt
                return d_inputs

        return outputs, grad_fn

    return _entmax_inner(inputs)


@tf.custom_gradient
def sparse_entmax15_loss_with_logits(labels, logits):
    """Computes sample-wise entmax1.5 loss :param labels: reference answers vector
    int64[batch_size] \in [0, num_classes) :param logits: output matrix
    float32[batch_size, num_classes] (not actually logits :) :returns: elementwise loss,
    float32[batch_size]"""
    assert logits.shape.ndims == 2 and labels.shape.ndims == 1
    with tf.name_scope("entmax_loss"):
        p_star = entmax15(logits, axis=-1)
        omega_entmax15 = (1 - (tf.reduce_sum(p_star * tf.sqrt(p_star), axis=-1))) / 0.75
        p_incr = p_star - tf.one_hot(labels, depth=tf.shape(logits)[-1], axis=-1)
        loss = omega_entmax15 + tf.einsum("ij,ij->i", p_incr, logits)

    def grad_fn(grad_output):
        with tf.name_scope("entmax_loss_grad"):
            return None, grad_output[..., None] * p_incr

    return loss, grad_fn


@tf.custom_gradient
def entmax15_loss_with_logits(labels, logits):
    """Computes sample-wise entmax1.5 loss :param logits: "logits" matrix
    float32[batch_size, num_classes] :param labels: reference answers indicators,
    float32[batch_size, num_classes] :returns: elementwise loss, float32[batch_size]

    WARNING: this function does not propagate gradients through :labels:
    This behavior is the same as like softmax_crossentropy_with_logits v1
    It may become an issue if you do something like co-distillation

    """
    assert labels.shape.ndims == logits.shape.ndims == 2
    with tf.name_scope("entmax_loss"):
        p_star = entmax15(logits, axis=-1)
        omega_entmax15 = (1 - (tf.reduce_sum(p_star * tf.sqrt(p_star), axis=-1))) / 0.75
        p_incr = p_star - labels
        loss = omega_entmax15 + tf.einsum("ij,ij->i", p_incr, logits)

    def grad_fn(grad_output):
        with tf.name_scope("entmax_loss_grad"):
            return None, grad_output[..., None] * p_incr

    return loss, grad_fn


def top_k_over_axis(inputs, k, axis=-1, **kwargs):
    """Performs tf.nn.top_k over any chosen axis."""
    with tf.name_scope("top_k_along_axis"):
        if axis == -1:
            return tf.nn.top_k(inputs, k, **kwargs)

        perm_order = list(range(inputs.shape.ndims))
        perm_order.append(perm_order.pop(axis))
        inv_order = [perm_order.index(i) for i in range(len(perm_order))]

        input_perm = tf.transpose(inputs, perm_order)
        input_perm_sorted, sort_indices_perm = tf.nn.top_k(input_perm, k=k, **kwargs)

        input_sorted = tf.transpose(input_perm_sorted, inv_order)
        sort_indices = tf.transpose(sort_indices_perm, inv_order)
    return input_sorted, sort_indices


def _make_ix_like(inputs, axis=-1):
    """Creates indices 0, ...

    , input[axis] unsqueezed to input dimensios

    """
    assert inputs.shape.ndims is not None
    rho = tf.cast(tf.range(1, tf.shape(inputs)[axis] + 1), dtype=inputs.dtype)
    view = [1] * inputs.shape.ndims
    view[axis] = -1
    return tf.reshape(rho, view)


def gather_over_axis(values, indices, gather_axis):
    """Replicates the behavior of torch.gather for tf<=1.8; for newer versions use
    tf.gather with batch_dims :param values: tensor [d0, ..., dn] :param indices: int64
    tensor of same shape as values except for gather_axis :param gather_axis: performs
    gather along this axis :returns: gathered values, same shape as values except for
    gather_axis If gather_axis == 2 gathered_values[i, j, k, ...] = values[i, j,
    indices[i, j, k, ...], ...] see torch.gather for more detils."""
    assert indices.shape.ndims is not None
    assert indices.shape.ndims == values.shape.ndims

    ndims = indices.shape.ndims
    gather_axis = gather_axis % ndims
    shape = tf.shape(indices)

    selectors = []
    for axis_i in range(ndims):
        if axis_i == gather_axis:
            selectors.append(indices)
        else:
            index_i = tf.range(
                tf.cast(shape[axis_i], dtype=indices.dtype), dtype=indices.dtype
            )
            index_i = tf.reshape(
                index_i, [-1 if i == axis_i else 1 for i in range(ndims)]
            )
            index_i = tf.tile(
                index_i, [shape[i] if i != axis_i else 1 for i in range(ndims)]
            )
            selectors.append(index_i)

    return tf.gather_nd(values, tf.stack(selectors, axis=-1))


def entmax_threshold_and_support(inputs, axis=-1):
    """Computes clipping threshold for entmax1.5 over specified axis NOTE this
    implementation uses the same heuristic as.

    the original code: https://tinyurl.com/pytorch-entmax-line-203
    :param inputs: (entmax1.5 inputs - max) / 2
    :param axis: entmax1.5 outputs will sum to 1 over this axis

    """

    with tf.name_scope("entmax_threshold_and_support"):
        num_outcomes = tf.shape(inputs)[axis]
        inputs_sorted, _ = top_k_over_axis(
            inputs, k=num_outcomes, axis=axis, sorted=True
        )

        rho = _make_ix_like(inputs, axis=axis)

        mean = tf.cumsum(inputs_sorted, axis=axis) / rho

        mean_sq = tf.cumsum(tf.square(inputs_sorted), axis=axis) / rho
        delta = (1 - rho * (mean_sq - tf.square(mean))) / rho

        delta_nz = tf.nn.relu(delta)
        tau = mean - tf.sqrt(delta_nz)

        support_size = tf.reduce_sum(
            tf.to_int64(tf.less_equal(tau, inputs_sorted)), axis=axis, keep_dims=True
        )

        tau_star = gather_over_axis(tau, support_size - 1, axis)
    return tau_star, support_size


class TabNetReduced(object):
    """Reduced TabNet model class."""

    def __init__(
        self,
        columns,
        num_features,
        feature_dim,
        output_dim,
        num_decision_steps,
        relaxation_factor,
        batch_momentum,
        virtual_batch_size,
        num_classes,
        epsilon=0.00001,
    ):
        """Initializes a reduced TabNet instance.

        Args:
          columns: The Tensorflow column names for the dataset.
          num_features: The number of input features (i.e the number of columns for
            tabular data assuming each feature is represented with 1 dimension).
          feature_dim: Dimensionality of the hidden representation in feature
            transformation block. Each layer first maps the representation to a
            2*feature_dim-dimensional output and half of it is used to determine the
            nonlinearity of the GLU activation where the other half is used as an
            input to GLU, and eventually feature_dim-dimensional output is
            transferred to the next layer.
          output_dim: Dimensionality of the outputs of each decision step, which is
            later mapped to the final classification or regression output.
          num_decision_steps: Number of sequential decision steps.
          relaxation_factor: Relaxation factor that promotes the reuse of each
            feature at different decision steps. When it is 1, a feature is enforced
            to be used only at one decision step and as it increases, more
            flexibility is provided to use a feature at multiple decision steps.
          batch_momentum: Momentum in ghost batch normalization.
          virtual_batch_size: Virtual batch size in ghost batch normalization. The
            overall batch size should be an integer multiple of virtual_batch_size.
          num_classes: Number of output classes.
          epsilon: A small number for numerical stability of the entropy calcations.

        Returns:
          A reduced TabNet instance.

        """

        self.columns = columns
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_classes = num_classes
        self.epsilon = epsilon

    def encoder(self, data, reuse, is_training):
        """Reduced TabNet encoder model."""

        with tf.variable_scope("Encoder", reuse=reuse):
            # Reads and normalizes input features.
            features = tf.feature_column.input_layer(data, self.columns)
            features = tf.layers.batch_normalization(
                features, training=is_training, momentum=self.batch_momentum
            )
            batch_size = tf.shape(features)[0]

            # Initializes decision-step dependent variables.
            output_aggregated = tf.zeros([batch_size, self.output_dim])
            masked_features = features
            mask_values = tf.zeros([batch_size, self.num_features])
            aggregated_mask_values = tf.zeros([batch_size, self.num_features])
            complementary_aggregated_mask_values = tf.ones(
                [batch_size, self.num_features]
            )
            total_entropy = 0

            v_b = self.virtual_batch_size if is_training else 1
            # Feature transformer: a sort of recurrent structure
            # TODO: can we automate number of decision steps needed?
            for ni in range(self.num_decision_steps):
                # Feature transformer with one shared and one decision step dependent
                # blocks is used below. This departs from the original model
                reuse_flag = ni > 0

                # shared because of the same name
                transform_f1 = tf.layers.dense(
                    masked_features,
                    self.feature_dim * 2,
                    name="Transform_f1",
                    reuse=reuse_flag,
                    use_bias=False,
                )
                transform_f1 = tf.layers.batch_normalization(
                    transform_f1,
                    training=is_training,
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b,
                )
                transform_f1 = glu(transform_f1, self.feature_dim)

                # step dependent
                transform_f2 = tf.layers.dense(
                    transform_f1,
                    self.feature_dim * 2,
                    name="Transform_f1" + str(ni),
                    use_bias=False,
                )
                transform_f2 = tf.layers.batch_normalization(
                    transform_f2,
                    training=is_training,
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b,
                )
                transform_f2 = (
                    glu(transform_f2, self.feature_dim) + transform_f1
                ) * np.sqrt(0.5)

                if ni > 0:
                    decision_out = tf.nn.relu(transform_f2[:, : self.output_dim])

                    # Decision aggregation.
                    output_aggregated += decision_out

                    # Aggregated masks are used for visualization of the
                    # feature importance attributes.
                    scale_agg = tf.reduce_sum(decision_out, axis=1, keep_dims=True) / (
                        self.num_decision_steps - 1
                    )
                    aggregated_mask_values += mask_values * scale_agg

                features_for_coef = transform_f2[:, self.output_dim :]

                # Attentive transformer
                if ni < self.num_decision_steps - 1:
                    # Determines the feature masks via linear and nonlinear
                    # transformations, taking into account of aggregated feature use.
                    mask_values = tf.layers.dense(
                        features_for_coef,
                        self.num_features,
                        name="Transform_coef" + str(ni),
                        use_bias=False,
                    )
                    mask_values = tf.layers.batch_normalization(
                        mask_values,
                        training=is_training,
                        momentum=self.batch_momentum,
                        virtual_batch_size=v_b,
                    )
                    mask_values *= complementary_aggregated_mask_values
                    # replace sparsemax with entmax 1.5
                    mask_values = entmax15(mask_values)

                    # Relaxation factor controls the amount of reuse of features between
                    # different decision blocks and updated with the values of
                    # coefficients.
                    complementary_aggregated_mask_values *= (
                        self.relaxation_factor - mask_values
                    )

                    # Entropy is used to penalize the amount of sparsity in feature
                    # selection.
                    total_entropy += tf.reduce_mean(
                        tf.reduce_sum(
                            -mask_values * tf.log(mask_values + self.epsilon), axis=1
                        )
                    ) / (self.num_decision_steps - 1)

                    # Feature selection.
                    masked_features = tf.multiply(mask_values, features)

                    # Visualization of the feature selection mask at decision step ni
                    tf.summary.image(
                        "Mask_for_step" + str(ni),
                        tf.expand_dims(tf.expand_dims(mask_values, 0), 3),
                        max_outputs=1,
                    )

            # Visualization of the aggregated feature importances
            tf.summary.image(
                "Aggregated_mask",
                tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3),
                max_outputs=1,
            )

            return output_aggregated, total_entropy

    def classify(self, activations, reuse):
        """Reduced TabNet classify block."""

        with tf.variable_scope("Classify", reuse=reuse):
            logits = tf.layers.dense(activations, self.num_classes, use_bias=False)
            predictions = tf.nn.softmax(logits)
            return logits, predictions

    def regress(self, activations, reuse):
        """Reduced TabNet regress block."""

        with tf.variable_scope("Regress", reuse=reuse):
            predictions = tf.layers.dense(activations, 1)
            return predictions
