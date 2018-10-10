
##>>>>>>>>>>>>>./pytorch2keras/converter.py======================================


"""
The Pytorch2Keras converter module over JIT-trace.
"""

import torch
import torch.jit
import torch.autograd
import torch.serialization
import contextlib
from torch.jit import _unique_state_dict

from .layers import AVAILABLE_CONVERTERS


@contextlib.contextmanager
def set_training(model, mode):
    """
    A context manager to temporarily set the training mode of 'model'
    to 'mode', resetting it when we exit the with-block.  A no-op if
    mode is None.
    """
    if mode is None:
        yield
        return
    old_mode = model.training
    if old_mode != mode:
        model.train(mode)
    try:
        yield
    finally:
        if old_mode != mode:
            model.train(old_mode)


def _optimize_graph(graph, aten):
    # run dce first to eliminate dead parts of the graph that might have been
    # left behind by things like symbolic_override
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_peephole(graph)
    torch._C._jit_pass_lint(graph)
    graph = torch._C._jit_pass_onnx(graph, aten)
    torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_onnx_peephole(graph)
    torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)
    graph = torch._C._jit_pass_canonicalize(graph)
    torch._C._jit_pass_lint(graph)
    return graph


def get_node_id(node):
    import re
    node_id = re.search(r"[\d]+", node.__str__())
    return node_id.group(0)


def pytorch_to_keras(
    model, args, input_shapes,
    change_ordering=False, training=False, verbose=False, short_names=False,
):
    """
    By given pytorch model convert layers with specified convertors.

    Args:
        model: pytorch model
        args: pytorch model arguments
        input_shapes: keras input shapes (using for each InputLayer)
        change_ordering: change CHW to HWC
        training: switch model to training mode
        verbose: verbose output
        short_names: use shorn names for keras layers

    Returns:
        model: created keras model.
    """

    # PyTorch JIT tracing
    if isinstance(args, torch.autograd.Variable):
        args = (args, )

    # Workaround for previous versions
    if isinstance(input_shapes, tuple):
        input_shapes = [input_shapes]

    orig_state_dict_keys = _unique_state_dict(model).keys()

    with set_training(model, training):
        trace, torch_out = torch.jit.get_trace_graph(model, tuple(args))

    if orig_state_dict_keys != _unique_state_dict(model).keys():
        raise RuntimeError("state_dict changed after running the tracer; "
                           "something weird is happening in your model!")

    # _optimize_trace(trace, False)
    trace.set_graph(_optimize_graph(trace.graph(), False))

    if verbose:
        print(trace.graph())

    if verbose:
        print(list(trace.graph().outputs()))

    # Get all graph nodes
    nodes = list(trace.graph().nodes())

    # Collect graph outputs
    graph_outputs = [n.uniqueName() for n in trace.graph().outputs()]
    print('Graph outputs:', graph_outputs)

    # Collect model state dict
    state_dict = _unique_state_dict(model)
    if verbose:
        print('State dict:', list(state_dict))

    import re
    import keras
    from keras import backend as K
    K.set_image_data_format('channels_first')

    layers = dict()
    keras_inputs = []
    for i in range(len(args)):
        layers['input{0}'.format(i)] = keras.layers.InputLayer(
            input_shape=input_shapes[i], name='input{0}'.format(i)
        ).output
        keras_inputs.append(layers['input{0}'.format(i)])

    outputs = []

    input_index = 0
    model_inputs = dict()
    for node in nodes:
        node_inputs = list(node.inputs())
        node_input_names = []
        for node_input in node_inputs:
            if node_input.node().scopeName():
                node_input_names.append(get_node_id(node_input.node()))

        if len(node_input_names) == 0:
            if len(node_inputs) > 0:
                if node_inputs[0] in model_inputs:
                    node_input_names.append(model_inputs[node_inputs[0]])
                else:
                    input_name = 'input{0}'.format(input_index)
                    node_input_names.append(input_name)
                    input_index += 1
                    model_inputs[node_inputs[0]] = input_name

        node_type = node.kind()
        # print(dir(node))

        node_scope_name = node.scopeName()
        node_id = get_node_id(node)
        node_weights_name = '.'.join(
            re.findall(r'\[([\w\d.]+)\]', node_scope_name)
        )
        node_attrs = {k: node[k] for k in node.attributeNames()}

        node_outputs = list(node.outputs())
        node_outputs_names = []
        for node_output in node_outputs:
            if node_output.node().scopeName():
                node_outputs_names.append(node_output.node().scopeName())

        if verbose:
            print(' ____ ')
            print('graph node:', node_scope_name)
            print('type:', node_type)
            print('inputs:', node_input_names)
            print('outputs:', node_outputs_names)
            print('name in state_dict:', node_weights_name)
            print('attrs:', node_attrs)
            print('is_terminal:', node_id in graph_outputs)
        AVAILABLE_CONVERTERS[node_type](
            node_attrs,
            node_weights_name, node_id,
            node_input_names,
            layers, state_dict,
            short_names
        )
        if node_id in graph_outputs:
            outputs.append(layers[node_id])

    model = keras.models.Model(inputs=keras_inputs, outputs=outputs)

    if change_ordering:
        import numpy as np
        conf = model.get_config()

        for layer in conf['layers']:
            if layer['config'] and 'batch_input_shape' in layer['config']:
                layer['config']['batch_input_shape'] = \
                    tuple(np.reshape(np.array(
                        [
                            [None] +
                            list(layer['config']['batch_input_shape'][2:][:]) +
                            [layer['config']['batch_input_shape'][1]]
                        ]), -1
                    ))
            if layer['config'] and 'target_shape' in layer['config']:
                if len(list(layer['config']['target_shape'][1:][:])) > 0:
                    layer['config']['target_shape'] = \
                        tuple(np.reshape(np.array(
                            [
                                list(layer['config']['target_shape'][1:][:]),
                                layer['config']['target_shape'][0]
                            ]), -1
                        ),)

            if layer['config'] and 'data_format' in layer['config']:
                layer['config']['data_format'] = 'channels_last'
            if layer['config'] and 'axis' in layer['config']:
                layer['config']['axis'] = 3

        K.set_image_data_format('channels_last')
        model_tf_ordering = keras.models.Model.from_config(conf)

        # from keras.utils.layer_utils import convert_all_kernels_in_model
        # convert_all_kernels_in_model(model)

        for dst_layer, src_layer in zip(
            model_tf_ordering.layers, model.layers
        ):
            dst_layer.set_weights(src_layer.get_weights())

        model = model_tf_ordering

    return model

##>>>>>>>>>>>>>./pytorch2keras/__init__.py======================================



##>>>>>>>>>>>>>./pytorch2keras/layers.py======================================


import keras.layers
import numpy as np
import random
import string
import tensorflow as tf


def random_string(length):
    """
    Generate a random string for the layer name.
    :param length: a length of required random string
    :return: generated random string
    """
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))


def convert_conv(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert convolution layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
        short_names: use short names
    """
    print('Converting convolution ...')

    if short_names:
        tf_name = 'C' + random_string(7)
    else:
        tf_name = w_name + str(random.random())

    bias_name = '{0}.bias'.format(w_name)
    weights_name = '{0}.weight'.format(w_name)
    input_name = inputs[0]

    if len(weights[weights_name].numpy().shape) == 5: # 3D conv
        W = weights[weights_name].numpy().transpose(2, 3, 4, 1, 0)
        height, width, channels, n_layers, n_filters = W.shape

        if bias_name in weights:
            biases = weights[bias_name].numpy()
            has_bias = True
        else:
            biases = None
            has_bias = False

        if params['pads'][0] > 0 or params['pads'][1] > 0:
            padding_name = tf_name + '_pad'
            padding_layer = keras.layers.ZeroPadding3D(
                padding=(params['pads'][0],
                         params['pads'][1],
                         params['pads'][2]),
                name=padding_name
            )
            layers[padding_name] = padding_layer(layers[input_name])
            input_name = padding_name

        if has_bias:
            weights = [W, biases]
        else:
            weights = [W]

        conv = keras.layers.Conv3D(
            filters=n_filters,
            kernel_size=(channels, height, width),
            strides=(params['strides'][0],
                     params['strides'][1],
                     params['strides'][2]),
            padding='valid',
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=params['dilations'][0],
            bias_initializer='zeros', kernel_initializer='zeros',
            name=tf_name
        )
        layers[scope_name] = conv(layers[input_name])
    elif len(weights[weights_name].numpy().shape) == 4:  # 2D conv
        if params['pads'][0] > 0 or params['pads'][1] > 0:
            padding_name = tf_name + '_pad'
            padding_layer = keras.layers.ZeroPadding2D(
                padding=(params['pads'][0], params['pads'][1]),
                name=padding_name
            )
            layers[padding_name] = padding_layer(layers[input_name])
            input_name = padding_name

        W = weights[weights_name].numpy().transpose(2, 3, 1, 0)
        height, width, channels_per_group, out_channels = W.shape
        n_groups = params['group']
        in_channels = channels_per_group * n_groups

        if n_groups == in_channels:
            print('Perform depthwise convolution: h={} w={} in={} out={}'
                .format(height, width, in_channels, out_channels))

            if bias_name in weights:
                biases = weights[bias_name].numpy()
                has_bias = True
            else:
                biases = None
                has_bias = False

            # We are just doing depthwise conv, so make the pointwise a no-op
            pointwise_wt = np.expand_dims(np.expand_dims(np.identity(out_channels), 0), 0)
            W = W.transpose(0, 1, 3, 2)
            if has_bias:
                weights = [W, pointwise_wt, biases]
            else:
                weights = [W, pointwise_wt]

            conv = keras.layers.SeparableConv2D(
                filters=out_channels,
                depth_multiplier=1,
                kernel_size=(height, width),
                strides=(params['strides'][0], params['strides'][1]),
                padding='valid',
                weights=weights,
                use_bias=has_bias,
                activation=None,
                bias_initializer='zeros', kernel_initializer='zeros',
                name=tf_name
            )
            layers[scope_name] = conv(layers[input_name])

        elif n_groups != 1:
            # Example from https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
            # # Split input and weights and convolve them separately
            # input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            # weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            # output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # # Concat the convolved output together again
            # conv = tf.concat(axis=3, values=output_groups)
            def target_layer(x, groups=params['group'], stride_y=params['strides'][0], stride_x=params['strides'][1]):
                x = tf.transpose(x, [0, 2, 3, 1])

                convolve = lambda i, k: tf.nn.conv2d(i, k,
                                                     strides=[1, stride_y, stride_x, 1],
                                                     padding='VALID')

                input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
                weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=W.transpose(0, 1, 2, 3))
                output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

                layer = tf.concat(axis=3, values=output_groups)

                layer = tf.transpose(layer, [0, 3, 1, 2])
                return layer

            lambda_layer = keras.layers.Lambda(target_layer)
            layers[scope_name] = lambda_layer(layers[input_name])

        else:
            if bias_name in weights:
                biases = weights[bias_name].numpy()
                has_bias = True
            else:
                biases = None
                has_bias = False

            if has_bias:
                weights = [W, biases]
            else:
                weights = [W]

            conv = keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=(height, width),
                strides=(params['strides'][0], params['strides'][1]),
                padding='valid',
                weights=weights,
                use_bias=has_bias,
                activation=None,
                dilation_rate=params['dilations'][0],
                bias_initializer='zeros', kernel_initializer='zeros',
                name=tf_name
            )
            layers[scope_name] = conv(layers[input_name])
    else:  # 1D conv
        W = weights[weights_name].numpy().transpose(2, 1, 0)
        width, channels, n_filters = W.shape

        if bias_name in weights:
            biases = weights[bias_name].numpy()
            has_bias = True
        else:
            biases = None
            has_bias = False

        padding_name = tf_name + '_pad'
        padding_layer = keras.layers.ZeroPadding1D(
            padding=params['pads'][0],
            name=padding_name
        )
        layers[padding_name] = padding_layer(layers[inputs[0]])
        input_name = padding_name

        if has_bias:
            weights = [W, biases]
        else:
            weights = [W]

        conv = keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=width,
            strides=params['strides'][0],
            padding='valid',
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=params['dilations'][0],
            bias_initializer='zeros', kernel_initializer='zeros',
            name=tf_name
        )
        layers[scope_name] = conv(layers[input_name])


def convert_convtranspose(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert transposed convolution layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting transposed convolution ...')

    if short_names:
        tf_name = 'C' + random_string(7)
    else:
        tf_name = w_name + str(random.random())

    bias_name = '{0}.bias'.format(w_name)
    weights_name = '{0}.weight'.format(w_name)

    if len(weights[weights_name].numpy().shape) == 4:
        W = weights[weights_name].numpy().transpose(2, 3, 1, 0)
        height, width, n_filters, channels = W.shape

        if bias_name in weights:
            biases = weights[bias_name].numpy()
            has_bias = True
        else:
            biases = None
            has_bias = False

        input_name = inputs[0]

        if has_bias:
            weights = [W, biases]
        else:
            weights = [W]

        conv = keras.layers.Conv2DTranspose(
            filters=n_filters,
            kernel_size=(height, width),
            strides=(params['strides'][0], params['strides'][1]),
            padding='valid',
            output_padding=0,
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=params['dilations'][0],
            bias_initializer='zeros', kernel_initializer='zeros',
            name=tf_name
        )

        layers[scope_name] = conv(layers[input_name])

        pads = params['pads']
        if pads[0] > 0:
            assert(len(pads) == 2 or (pads[2] == pads[0] and pads[3] == pads[1]))

            crop = keras.layers.Cropping2D(
                pads[:2],
                name=tf_name + '_crop'
            )
            layers[scope_name] = crop(layers[scope_name])
    else:
        raise AssertionError('Layer is not supported for now')


def convert_flatten(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert reshape(view).

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting flatten ...')

    if short_names:
        tf_name = 'R' + random_string(7)
    else:
        tf_name = w_name + str(random.random())

    reshape = keras.layers.Reshape([-1], name=tf_name)
    layers[scope_name] = reshape(layers[inputs[0]])


def convert_gemm(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert Linear.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting Linear ...')

    if short_names:
        tf_name = 'FC' + random_string(6)
    else:
        tf_name = w_name + str(random.random())

    bias_name = '{0}.bias'.format(w_name)
    weights_name = '{0}.weight'.format(w_name)

    W = weights[weights_name].numpy().transpose()
    input_channels, output_channels = W.shape

    keras_weights = [W]
    has_bias = False
    if bias_name in weights:
        bias = weights[bias_name].numpy()
        keras_weights = [W, bias]
        has_bias = True

    dense = keras.layers.Dense(
        output_channels,
        weights=keras_weights, use_bias=has_bias, name=tf_name, bias_initializer='zeros', kernel_initializer='zeros',
    )

    layers[scope_name] = dense(layers[inputs[0]])


def convert_avgpool(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert Average pooling.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting pooling ...')

    if short_names:
        tf_name = 'P' + random_string(7)
    else:
        tf_name = w_name + str(random.random())

    height, width = params['kernel_shape']
    stride_height, stride_width = params['strides']
    padding_h, padding_w, _, _ = params['pads']

    input_name = inputs[0]
    padding = 'valid'
    if padding_h > 0 and padding_w > 0:
        if padding_h == height // 2 and padding_w == width // 2:
            padding = 'same'
        else:
            raise AssertionError('Custom padding isnt supported')

    pooling = keras.layers.AveragePooling2D(
        pool_size=(height, width),
        strides=(stride_height, stride_width),
        padding=padding,
        name=tf_name
    )

    layers[scope_name] = pooling(layers[input_name])


def convert_maxpool(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert Max pooling.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """

    print('Converting pooling ...')

    if short_names:
        tf_name = 'P' + random_string(7)
    else:
        tf_name = w_name + str(random.random())

    if 'kernel_shape' in params:
        height, width = params['kernel_shape']
    else:
        height, width = params['kernel_size']

    if 'strides' in params:
        stride_height, stride_width = params['strides']
    else:
        stride_height, stride_width = params['stride']

    if 'pads' in params:
        padding_h, padding_w, _, _ = params['pads']
    else:
        padding_h, padding_w = params['padding']

    input_name = inputs[0]
    if padding_h > 0 and padding_w > 0:
        padding_name = tf_name + '_pad'
        padding_layer = keras.layers.ZeroPadding2D(
            padding=(padding_h, padding_w),
            name=padding_name
        )
        layers[padding_name] = padding_layer(layers[inputs[0]])
        input_name = padding_name

    # Pooling type
    pooling = keras.layers.MaxPooling2D(
        pool_size=(height, width),
        strides=(stride_height, stride_width),
        padding='valid',
        name=tf_name
    )

    layers[scope_name] = pooling(layers[input_name])


def convert_maxpool3(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert 3d Max pooling.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """

    print('Converting pooling ...')

    if short_names:
        tf_name = 'P' + random_string(7)
    else:
        tf_name = w_name + str(random.random())

    if 'kernel_shape' in params:
        height, width, depth = params['kernel_shape']
    else:
        height, width, depth = params['kernel_size']

    if 'strides' in params:
        stride_height, stride_width, stride_depth = params['strides']
    else:
        stride_height, stride_width, stride_depth = params['stride']

    if 'pads' in params:
        padding_h, padding_w, padding_d, _, _ = params['pads']
    else:
        padding_h, padding_w, padding_d = params['padding']

    input_name = inputs[0]
    if padding_h > 0 and padding_w > 0 and padding_d > 0:
        padding_name = tf_name + '_pad'
        padding_layer = keras.layers.ZeroPadding3D(
            padding=(padding_h, padding_w, padding_d),
            name=padding_name
        )
        layers[padding_name] = padding_layer(layers[inputs[0]])
        input_name = padding_name

    # Pooling type
    pooling = keras.layers.MaxPooling3D(
        pool_size=(height, width, depth),
        strides=(stride_height, stride_width, stride_depth),
        padding='valid',
        name=tf_name
    )

    layers[scope_name] = pooling(layers[input_name])


def convert_dropout(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert dropout.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting dropout ...')

    if short_names:
        tf_name = 'DO' + random_string(6)
    else:
        tf_name = w_name + str(random.random())

    dropout = keras.layers.Dropout(rate=params['ratio'], name=tf_name)
    layers[scope_name] = dropout(layers[inputs[0]])


def convert_batchnorm(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert batch normalization layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting batchnorm ...')

    if short_names:
        tf_name = 'BN' + random_string(6)
    else:
        tf_name = w_name + str(random.random())

    bias_name = '{0}.bias'.format(w_name)
    weights_name = '{0}.weight'.format(w_name)
    mean_name = '{0}.running_mean'.format(w_name)
    var_name = '{0}.running_var'.format(w_name)

    if bias_name in weights:
        beta = weights[bias_name].numpy()

    if weights_name in weights:
        gamma = weights[weights_name].numpy()

    mean = weights[mean_name].numpy()
    variance = weights[var_name].numpy()

    eps = params['epsilon']
    momentum = params['momentum']

    if weights_name not in weights:
        bn = keras.layers.BatchNormalization(
            axis=1, momentum=momentum, epsilon=eps,
            center=False, scale=False,
            weights=[mean, variance],
            name=tf_name
        )
    else:
        bn = keras.layers.BatchNormalization(
            axis=1, momentum=momentum, epsilon=eps,
            weights=[gamma, beta, mean, variance],
            name=tf_name
        )
    layers[scope_name] = bn(layers[inputs[0]])


def convert_instancenorm(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert instance normalization layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting instancenorm ...')

    if short_names:
        tf_name = 'IN' + random_string(6)
    else:
        tf_name = w_name + str(random.random())

    assert(len(inputs) == 3)

    gamma = layers[inputs[-2]]
    beta = layers[inputs[-1]]

    def target_layer(x, epsilon=params['epsilon'], gamma=gamma, beta=beta):
        layer = tf.contrib.layers.instance_norm(x,
            param_initializers={'beta': tf.constant_initializer(beta), 'gamma': tf.constant_initializer(gamma)},
            epsilon=epsilon, data_format='NCHW',
            trainable=False)
        return layer

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers[inputs[0]])


def convert_elementwise_add(
    params, w_name, scope_name, inputs, layers, weights, short_names
):
    """
    Convert elementwise addition.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting elementwise_add ...')
    model0 = layers[inputs[0]]
    model1 = layers[inputs[1]]

    if short_names:
        tf_name = 'A' + random_string(7)
    else:
        tf_name = w_name + str(random.random())

    add = keras.layers.Add(name=tf_name)
    layers[scope_name] = add([model0, model1])


def convert_elementwise_mul(
    params, w_name, scope_name, inputs, layers, weights, short_names
):
    """
    Convert elementwise multiplication.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting elementwise_mul ...')
    model0 = layers[inputs[0]]
    model1 = layers[inputs[1]]

    if short_names:
        tf_name = 'M' + random_string(7)
    else:
        tf_name = w_name + str(random.random())

    mul = keras.layers.Multiply(name=tf_name)
    layers[scope_name] = mul([model0, model1])


def convert_elementwise_sub(
    params, w_name, scope_name, inputs, layers, weights, short_names
):
    """
    Convert elementwise subtraction.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting elementwise_sub ...')
    model0 = layers[inputs[0]]
    model1 = layers[inputs[1]]

    if short_names:
        tf_name = 'S' + random_string(7)
    else:
        tf_name = w_name + str(random.random())

    sub = keras.layers.Subtract(name=tf_name)
    layers[scope_name] = sub([model0, model1])


def convert_sum(
    params, w_name, scope_name, inputs, layers, weights, short_names
):
    """
    Convert sum.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting Sum ...')

    def target_layer(x):
        import keras.backend as K
        return K.sum(x)

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers[inputs[0]])


def convert_concat(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert concatenation.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting concat ...')
    concat_nodes = [layers[i] for i in inputs]

    if len(concat_nodes) == 1:
        # no-op
        layers[scope_name] = concat_nodes[0]
        return

    if short_names:
        tf_name = 'CAT' + random_string(5)
    else:
        tf_name = w_name + str(random.random())

    cat = keras.layers.Concatenate(name=tf_name, axis=params['axis'])
    layers[scope_name] = cat(concat_nodes)


def convert_relu(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert relu layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting relu ...')

    if short_names:
        tf_name = 'RELU' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    relu = keras.layers.Activation('relu', name=tf_name)
    layers[scope_name] = relu(layers[inputs[0]])


def convert_lrelu(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert leaky relu layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting lrelu ...')

    if short_names:
        tf_name = 'lRELU' + random_string(3)
    else:
        tf_name = w_name + str(random.random())

    leakyrelu = \
        keras.layers.LeakyReLU(alpha=params['alpha'], name=tf_name)
    layers[scope_name] = leakyrelu(layers[inputs[0]])


def convert_sigmoid(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert sigmoid layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting sigmoid ...')

    if short_names:
        tf_name = 'SIGM' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    sigmoid = keras.layers.Activation('sigmoid', name=tf_name)
    layers[scope_name] = sigmoid(layers[inputs[0]])


def convert_softmax(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert softmax layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting softmax ...')

    if short_names:
        tf_name = 'SMAX' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    softmax = keras.layers.Activation('softmax', name=tf_name)
    layers[scope_name] = softmax(layers[inputs[0]])


def convert_tanh(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert tanh layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting tanh ...')

    if short_names:
        tf_name = 'TANH' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    tanh = keras.layers.Activation('tanh', name=tf_name)
    layers[scope_name] = tanh(layers[inputs[0]])


def convert_hardtanh(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert hardtanh layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting hardtanh (clip) ...')

    def target_layer(x, max_val=float(params['max_val']), min_val=float(params['min_val'])):
        return tf.minimum(max_val, tf.maximum(min_val, x))

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers[inputs[0]])


def convert_selu(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert selu layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting selu ...')

    if short_names:
        tf_name = 'SELU' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    selu = keras.layers.Activation('selu', name=tf_name)
    layers[scope_name] = selu(layers[inputs[0]])


def convert_transpose(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert transpose layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting transpose ...')
    if params['perm'][0] != 0:
        # raise AssertionError('Cannot permute batch dimension')
        print('!!! Cannot permute batch dimension. Result may be wrong !!!')
        try:
            layers[scope_name] = layers[inputs[0]]
        except:
            pass
    else:
        if short_names:
            tf_name = 'PERM' + random_string(4)
        else:
            tf_name = w_name + str(random.random())
        permute = keras.layers.Permute(params['perm'][1:], name=tf_name)
        layers[scope_name] = permute(layers[inputs[0]])


def convert_reshape(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert reshape layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting reshape ...')
    if short_names:
        tf_name = 'RESH' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    if len(inputs) > 1:
        if layers[inputs[1]][0] == -1:
            print('Cannot deduct batch size! It will be omitted, but result may be wrong.')

        reshape = keras.layers.Reshape(layers[inputs[1]][1:], name=tf_name)
        layers[scope_name] = reshape(layers[inputs[0]])
    else:
        reshape = keras.layers.Reshape(params['shape'][1:], name=tf_name)
        layers[scope_name] = reshape(layers[inputs[0]])


def convert_matmul(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert matmul layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting matmul ...')

    if short_names:
        tf_name = 'MMUL' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    if len(inputs) == 1:
        weights_name = '{0}.weight'.format(w_name)

        W = weights[weights_name].numpy().transpose()
        input_channels, output_channels = W.shape

        keras_weights = [W]

        dense = keras.layers.Dense(
            output_channels,
            weights=keras_weights, use_bias=False, name=tf_name, bias_initializer='zeros', kernel_initializer='zeros',
        )
        layers[scope_name] = dense(layers[inputs[0]])
    elif len(inputs) == 2:
        weights_name = '{0}.weight'.format(w_name)

        W = weights[weights_name].numpy().transpose()
        input_channels, output_channels = W.shape

        keras_weights = [W]

        dense = keras.layers.Dense(
            output_channels,
            weights=keras_weights, use_bias=False, name=tf_name, bias_initializer='zeros', kernel_initializer='zeros',
        )
        layers[scope_name] = dense(layers[inputs[0]])
    else:
        raise AssertionError('Cannot convert matmul layer')


def convert_gather(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert gather (embedding) layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting embedding ...')

    if short_names:
        tf_name = 'EMBD' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    weights_name = '{0}.weight'.format(w_name)

    W = weights[weights_name].numpy()
    input_channels, output_channels = W.shape

    keras_weights = [W]

    dense = keras.layers.Embedding(
        input_channels,
        weights=keras_weights, output_dim=output_channels, name=tf_name
    )
    layers[scope_name] = dense(layers[inputs[0]])


def convert_reduce_sum(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert reduce_sum layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting reduce_sum ...')

    keepdims = params['keepdims'] > 0
    axis = params['axes']

    def target_layer(x, keepdims=keepdims, axis=axis):
        import keras.backend as K
        return K.sum(x, keepdims=keepdims, axis=axis)

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers[inputs[0]])


def convert_constant(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert constant layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting constant ...')

    # params_list = params['value'].numpy().tolist()

    # def target_layer(x):
    #     import keras.backend as K
    #     return K.constant(params_list)

    # lambda_layer = keras.layers.Lambda(target_layer)
    # layers[scope_name] = lambda_layer(layers['input0']) # Temporary fix for nonexistent input name created by converter.py
    layers[scope_name] = params['value'].tolist()


def convert_upsample(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert upsample_bilinear2d layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting upsample...')

    if params['mode'] != 'nearest':
        raise AssertionError('Cannot convert non-nearest upsampling')

    if short_names:
        tf_name = 'UPSL' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    scale = (params['height_scale'], params['width_scale'])
    upsampling = keras.layers.UpSampling2D(
        size=scale, name=tf_name
    )
    layers[scope_name] = upsampling(layers[inputs[0]])


def convert_padding(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert padding layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting padding...')

    if params['mode'] == 'constant':
        # raise AssertionError('Cannot convert non-constant padding')

        if params['value'] != 0.0:
            raise AssertionError('Cannot convert non-zero padding')

        if short_names:
            tf_name = 'PADD' + random_string(4)
        else:
            tf_name = w_name + str(random.random())

        # Magic ordering
        padding_name = tf_name
        padding_layer = keras.layers.ZeroPadding2D(
            padding=((params['pads'][2], params['pads'][6]), (params['pads'][3], params['pads'][7])),
            name=padding_name
        )

        layers[scope_name] = padding_layer(layers[inputs[0]])
    elif params['mode'] == 'reflect':

        def target_layer(x, pads=params['pads']):
            # x = tf.transpose(x, [0, 2, 3, 1])
            layer = tf.pad(x, [[0, 0], [0, 0], [pads[2], pads[6]], [pads[3], pads[7]]], 'REFLECT')
            # layer = tf.transpose(layer, [0, 3, 1, 2])
            return layer

        lambda_layer = keras.layers.Lambda(target_layer)
        layers[scope_name] = lambda_layer(layers[inputs[0]])


def convert_adaptive_avg_pool2d(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert adaptive_avg_pool2d layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting adaptive_avg_pool2d...')

    if short_names:
        tf_name = 'APOL' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    global_pool = keras.layers.GlobalAveragePooling2D()
    layers_global_pool = global_pool(layers[inputs[0]])

    def target_layer(x):
        return keras.backend.expand_dims(x)

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers_global_pool)


def convert_slice(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert slice operation.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting slice ...')

    if len(params['axes']) > 1:
        raise AssertionError('Cannot convert slice by multiple dimensions')

    if params['axes'][0] not in [0, 1, 2, 3]:
        raise AssertionError('Slice by dimension more than 3 or less than 0 is not supported')

    def target_layer(x, axis=int(params['axes'][0]), start=int(params['starts'][0]), end=int(params['ends'][0])):
        if axis == 0:
            return x[start:end]
        elif axis == 1:
            return x[:, start:end]
        elif axis == 2:
            return x[:, :, start:end]
        elif axis == 3:
            return x[:, :, :, start:end]

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers[inputs[0]])


def convert_squeeze(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert squeeze operation.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting squeeze ...')

    if len(params['axes']) > 1:
        raise AssertionError('Cannot convert squeeze by multiple dimensions')

    def target_layer(x, axis=int(params['axes'][0])):
        return tf.squeeze(x, axis=axis)

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers[inputs[0]])


AVAILABLE_CONVERTERS = {
    'onnx::Conv': convert_conv,
    'onnx::ConvTranspose': convert_convtranspose,
    'onnx::Flatten': convert_flatten,
    'onnx::Gemm': convert_gemm,
    'onnx::MaxPool': convert_maxpool,
    'max_pool2d': convert_maxpool,
    'aten::max_pool3d': convert_maxpool3,
    'aten::max_pool2d': convert_maxpool,
    'onnx::AveragePool': convert_avgpool,
    'onnx::Dropout': convert_dropout,
    'onnx::BatchNormalization': convert_batchnorm,
    'onnx::InstanceNormalization': convert_instancenorm,
    'onnx::Add': convert_elementwise_add,
    'onnx::Mul': convert_elementwise_mul,
    'onnx::Sub': convert_elementwise_sub,
    'onnx::Sum': convert_sum,
    'onnx::Concat': convert_concat,
    'onnx::Relu': convert_relu,
    'onnx::LeakyRelu': convert_lrelu,
    'onnx::Sigmoid': convert_sigmoid,
    'onnx::Softmax': convert_softmax,
    'onnx::Tanh': convert_tanh,
    'aten::hardtanh': convert_hardtanh,
    'onnx::Selu': convert_selu,
    'onnx::Transpose': convert_transpose,
    'onnx::Reshape': convert_reshape,
    'onnx::MatMul': convert_matmul,
    'onnx::Gather': convert_gather,
    'onnx::ReduceSum': convert_reduce_sum,
    'onnx::Constant': convert_constant,
    'onnx::Upsample': convert_upsample,
    'onnx::Pad': convert_padding,
    'aten::adaptive_avg_pool2d': convert_adaptive_avg_pool2d,
    'onnx::Slice': convert_slice,
    'onnx::Squeeze': convert_squeeze,
}

##>>>>>>>>>>>>>./tests/layers/max_pool.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class MaxPool(nn.Module):
    """Module for MaxPool conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(MaxPool, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)
        self.pool = nn.MaxPool2d(kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.pool(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 100)
        out = np.random.randint(1, 100)

        model = MaxPool(inp, out, kernel_size, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/depthwise_conv2d.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


def depthwise_conv3x3(channels,
                      stride):
    return nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=channels,
        bias=False)


class TestConv2d(nn.Module):
    """Module for Conv2d conversion testing
    """

    def __init__(self, inp=10, stride=1):
        super(TestConv2d, self).__init__()
        self.conv2d_dw = depthwise_conv3x3(inp, stride)

    def forward(self, x):
        x = self.conv2d_dw(x)
        return x


def check_error(output, k_model, input_np, epsilon=1e-5):
    pytorch_output = output.data.numpy()
    keras_output = k_model.predict(input_np)

    error = np.max(pytorch_output - keras_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 100)
        stride = np.random.randint(1, 3)

        model = TestConv2d(inp, stride)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        error = check_error(output, k_model, input_np)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/tanh.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestTanh(nn.Module):
    """Module for Tanh activation conversion testing
    """

    def __init__(self, inp=10, out=16, bias=True):
        super(TestTanh, self).__init__()
        self.linear = nn.Linear(inp, out, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.tanh(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        inp = np.random.randint(1, 100)
        out = np.random.randint(1, 100)
        model = TestTanh(inp, out, inp % 2)

        input_np = np.random.uniform(-1.0, 1.0, (1, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/slice.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestSlice(nn.Module):
    """Module for Slicings conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestSlice, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)

    def forward(self, x):
        x = self.conv2d(x)
        return x[:, 0, :, :]


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 100)
        out = np.random.randint(1, 100)

        model = TestSlice(inp, out, kernel_size, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/conv2d_dilation.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestConv2d(nn.Module):
    """Module for Conv2d conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, dilation=1, bias=True):
        super(TestConv2d, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias, dilation=dilation)

    def forward(self, x):
        x = self.conv2d(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        dilation = np.random.randint(1, kernel_size + 1)
        inp = np.random.randint(kernel_size + 1, 100)
        out = np.random.randint(1, 100)

        model = TestConv2d(inp, out, kernel_size, dilation, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/embedding.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestEmbedding(nn.Module):
    def __init__(self, input_size):
        super(TestEmbedding, self).__init__()
        self.embedd = nn.Embedding(input_size, 100)

    def forward(self, input):
        return self.embedd(input)


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        input_np = np.random.randint(0, 10, (1, 1, 4))
        input = Variable(torch.LongTensor(input_np))

        simple_net = TestEmbedding(1000)
        output = simple_net(input)

        k_model = pytorch_to_keras(simple_net, input, (1, 4), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output[0])
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/dense.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestDense(nn.Module):
    """Module for Dense conversion testing
    """

    def __init__(self, inp=10, out=16, bias=True):
        super(TestDense, self).__init__()
        self.linear = nn.Linear(inp, out, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        inp = np.random.randint(1, 100)
        out = np.random.randint(1, 100)
        model = TestDense(inp, out, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp))
        input_var = Variable(torch.FloatTensor(input_np))

        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/conv2d_channels_last.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestConv2d(nn.Module):
    """Module for Conv2d conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestConv2d, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, stride=1, kernel_size=kernel_size, bias=bias)

    def forward(self, x):
        x = self.conv2d(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 10)
        inp = np.random.randint(kernel_size + 1, 100)
        out = np.random.randint(1, 2)

        model = TestConv2d(inp + 2, out, kernel_size, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp + 2, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp + 2, inp, inp,), change_ordering=True, verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np.transpose(0, 2, 3, 1))

        error = np.max(pytorch_output - keras_output.transpose(0, 3, 1, 2))
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/densenet.py======================================


import numpy as np
import torch
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision


if __name__ == '__main__':
    max_error = 0
    for i in range(10):
        model = torchvision.models.DenseNet()
        for m in model.modules():
            m.training = False

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (3, 224, 224,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/droupout.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestDropout(nn.Module):
    """Module for Dropout conversion testing
    """

    def __init__(self, inp=10, out=16, p=0.5, bias=True):
        super(TestDropout, self).__init__()
        self.linear = nn.Linear(inp, out, bias=bias)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        inp = np.random.randint(1, 100)
        out = np.random.randint(1, 100)
        p = np.random.uniform(0, 1)
        model = TestDropout(inp, out, inp % 2, p)
        model.eval()

        input_np = np.random.uniform(-1.0, 1.0, (1, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp,), verbose=True)

        keras_output = k_model.predict(input_np)

        pytorch_output = output.data.numpy()

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

        # not implemented yet

##>>>>>>>>>>>>>./tests/layers/__init__.py======================================



##>>>>>>>>>>>>>./tests/layers/mul.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestMul(nn.Module):
    """Module for Element-wise multiplication conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestMul, self).__init__()
        self.conv2d_1 = nn.Conv2d(inp, out, stride=(inp % 3 + 1), kernel_size=kernel_size, bias=bias)
        self.conv2d_2 = nn.Conv2d(inp, out, stride=(inp % 3 + 1), kernel_size=kernel_size, bias=bias)

    def forward(self, x):
        x1 = self.conv2d_1(x)
        x2 = self.conv2d_2(x)
        return (x1 * x2).sum()


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 100)
        out = np.random.randint(1, 100)

        model = TestMul(inp, out, kernel_size, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/instance_norm.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestInstanceNorm2d(nn.Module):
    """Module for InstanceNorm2d conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestInstanceNorm2d, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)
        self.bn = nn.InstanceNorm2d(out, affine=True)
        self.bn.weight = torch.nn.Parameter(torch.FloatTensor(self.bn.weight.size()).uniform_(0,1))
        self.bn.bias = torch.nn.Parameter(torch.FloatTensor(self.bn.bias.size()).uniform_(2,3))

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 100)
        out = np.random.randint(1, 100)

        model = TestInstanceNorm2d(inp, out, kernel_size, inp % 2)
        model.eval()
        for m in model.modules():
            m.training = False

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/group_conv2d.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


def group_conv1x1(in_channels,
                  out_channels,
                  groups):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
        groups=groups,
        bias=False)


class TestGroupConv2d(nn.Module):
    """Module for Conv2d conversion testing
    """

    def __init__(self, inp=10, groups=1):
        super(TestGroupConv2d, self).__init__()
        self.conv2d_group = group_conv1x1(inp, inp, groups)

    def forward(self, x):
        x = self.conv2d_group(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        groups = np.random.randint(1, 10)
        inp = np.random.randint(kernel_size + 1, 10)  * groups
        h, w = 32, 32
        model = TestGroupConv2d(inp, groups)

        input_np = np.random.uniform(0, 1, (1, inp, h, w))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, h, w,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/conv3d.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestConv3d(nn.Module):
    """Module for Conv2d conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestConv3d, self).__init__()
        self.conv3d = nn.Conv3d(inp, out, kernel_size=kernel_size, bias=bias)

    def forward(self, x):
        x = self.conv3d(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 30)
        out = np.random.randint(1, 30)

        model = TestConv3d(inp, out, kernel_size, inp % 2)

        input_var = Variable(torch.randn(1, inp, inp, inp, inp))

        output = model(input_var)

        k_model = pytorch_to_keras(model,
                                   input_var,
                                   (inp, inp, inp, inp,),
                                   verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_var.numpy())
        error = np.max(pytorch_output - keras_output)
        print("iteration: {}, error: {}".format(i, error))
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/view.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestView(nn.Module):
    """Module for View conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestView, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)

    def forward(self, x):
        x = self.conv2d(x)
        x = x.view([x.size(0), -1, 2, 1, 1, 1, 1, 1]).view(x.size(0), -1).view(x.size(0), -1)
        x = torch.nn.Tanh()(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = 2 * np.random.randint(kernel_size + 1, 10)
        out = 2 * np.random.randint(1, 10)

        model = TestView(inp, out, kernel_size, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/transpose.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestTranspose(nn.Module):
    """Module for Transpose conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestTranspose, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)

    def forward(self, x):
        x = self.conv2d(x)
        x = torch.transpose(x, 2, 3)
        x = torch.nn.Tanh()(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 100)
        out = np.random.randint(1, 100)

        model = TestTranspose(inp, out, kernel_size, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/conv2d.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestConv2d(nn.Module):
    """Module for Conv2d conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestConv2d, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)

    def forward(self, x):
        x = self.conv2d(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 100)
        out = np.random.randint(1, 100)

        model = TestConv2d(inp, out, kernel_size, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/avg_pool.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class AvgPool(nn.Module):
    """Module for MaxPool conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(AvgPool, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, padding=3, bias=bias)
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, count_include_pad=True)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.pool(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 100)
        out = np.random.randint(1, 100)

        model = AvgPool(inp, out, kernel_size, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/channel_shuffle.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


def channel_shuffle(x, groups):
    """Channel Shuffle operation from ShuffleNet [arxiv: 1707.01083]
    Arguments:
        x (Tensor): tensor to shuffle.
        groups (int): groups to be split
    """
    batch, channels, height, width = x.size()
    #assert (channels % groups == 0)
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x


class TestChannelShuffle2d(nn.Module):
    """Module for Channel shuffle conversion testing
    """

    def __init__(self, inp=10, out=16, groups=32):
        super(TestChannelShuffle2d, self).__init__()
        self.groups = groups
        self.conv2d = nn.Conv2d(inp, out, kernel_size=3, bias=False)

    def forward(self, x):
        x = self.conv2d(x)
        x = channel_shuffle(x, self.groups)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        groups = np.random.randint(1, 32)
        inp = np.random.randint(3, 32)
        out = np.random.randint(3, 32) * groups

        model = TestChannelShuffle2d(inp, out, groups)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))


##>>>>>>>>>>>>>./tests/layers/upsample_nearest.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torch.nn.functional as F


class TestUpsampleNearest2d(nn.Module):
    """Module for UpsampleNearest2d conversion testing
    """
    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestUpsampleNearest2d, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        x = self.conv2d(x)
        x = F.upsample(x, scale_factor=2)
        x = self.up(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 100)
        out = np.random.randint(1, 100)

        model = TestUpsampleNearest2d(inp, out, kernel_size, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/relu.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestRelu(nn.Module):
    """Module for ReLu conversion testing
    """

    def __init__(self, inp=10, out=16, bias=True):
        super(TestRelu, self).__init__()
        self.linear = nn.Linear(inp, out, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        inp = np.random.randint(1, 100)
        out = np.random.randint(1, 100)
        model = TestRelu(inp, out, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/sub.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestSub(nn.Module):
    """Module for Element-wise subtaction conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestSub, self).__init__()
        self.conv2d_1 = nn.Conv2d(inp, out, stride=(inp % 3 + 1), kernel_size=kernel_size, bias=bias)
        self.conv2d_2 = nn.Conv2d(inp, out, stride=(inp % 3 + 1), kernel_size=kernel_size, bias=bias)

    def forward(self, x):
        x1 = self.conv2d_1(x)
        x2 = self.conv2d_2(x)
        return x1 - x2


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 100)
        out = np.random.randint(1, 100)

        model = TestSub(inp, out, kernel_size, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/bn.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestConv2d(nn.Module):
    """Module for BatchNorm2d conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestConv2d, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)
        self.bn = nn.BatchNorm2d(out)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 100)
        out = np.random.randint(1, 100)

        model = TestConv2d(inp, out, kernel_size, inp % 2)
        for m in model.modules():
            m.training = False

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/sum.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestSum(nn.Module):
    def __init__(self, input_size):
        super(TestSum, self).__init__()
        self.embedd = nn.Embedding(input_size, 100)

    def forward(self, input):
        return self.embedd(input).sum(dim=0)


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        input_np = np.random.randint(0, 10, (1, 1, 4))
        input = Variable(torch.LongTensor(input_np))

        simple_net = TestSum(1000)
        output = simple_net(input)

        k_model = pytorch_to_keras(simple_net, input, (1, 4), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output[0])
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/multiple_inputs.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestMultipleInputs(nn.Module):
    """Module for Conv2d conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestMultipleInputs, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)

    def forward(self, x, y, z):
        return self.conv2d(x) + self.conv2d(y) + self.conv2d(z)


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 100)
        out = np.random.randint(1, 100)

        model = TestMultipleInputs(inp, out, kernel_size, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        input_var2 = Variable(torch.FloatTensor(input_np))
        input_var3 = Variable(torch.FloatTensor(input_np))
        output = model(input_var, input_var2, input_var3)

        k_model = pytorch_to_keras(model, [input_var, input_var2, input_var3], [(inp, inp, inp,), (inp, inp, inp,), (inp, inp, inp,)], verbose=True)
        k_model.summary()
        pytorch_output = output.data.numpy()
        keras_output = k_model.predict([input_np, input_np, input_np])

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/max_pool3d.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class MaxPool(nn.Module):
    """Module for MaxPool conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(MaxPool, self).__init__()
        self.conv3d = nn.Conv3d(inp, out, kernel_size=kernel_size, bias=bias)
        self.pool3d = nn.MaxPool3d(kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.pool3d(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 30)
        out = np.random.randint(1, 30)

        model = MaxPool(inp, out, kernel_size, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp, inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/sigmoid.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestSigmoid(nn.Module):
    """Module for Sigmoid activation conversion testing
    """

    def __init__(self, inp=10, out=16, bias=True):
        super(TestSigmoid, self).__init__()
        self.linear = nn.Linear(inp, out, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        inp = np.random.randint(1, 100)
        out = np.random.randint(1, 100)
        model = TestSigmoid(inp, out, inp % 2)

        input_np = np.random.uniform(-1.0, 1.0, (1, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/convtranspose2d.py======================================


import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestConvTranspose2d(nn.Module):
    """Module for ConvTranspose2d conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, stride=1, bias=True, padding=0):
        super(TestConvTranspose2d, self).__init__()
        self.conv2d = nn.ConvTranspose2d(inp, out, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv2d(x)
        return x

class ConvTranspose2dTest(unittest.TestCase):
    N = 100

    def test(self):
        max_error = 0
        for i in range(self.N):
            kernel_size = np.random.randint(1, 7)
            inp = np.random.randint(kernel_size + 1, 100)
            out = np.random.randint(1, 100)

            model = TestConvTranspose2d(inp, out, kernel_size, 2, inp % 3)

            input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
            input_var = Variable(torch.FloatTensor(input_np))
            output = model(input_var)

            k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

            pytorch_output = output.data.numpy()
            keras_output = k_model.predict(input_np)

            error = np.max(pytorch_output - keras_output)
            print(error)
            if max_error < error:
                max_error = error

        print('Max error: {0}'.format(max_error))

    def test_with_padding(self):
        max_error = 0
        for i in range(self.N):
            kernel_size = np.random.randint(1, 7)
            inp = np.random.randint(kernel_size + 1, 100)
            out = np.random.randint(1, 100)

            model = TestConvTranspose2d(inp, out, kernel_size, 2, inp % 3, padding=1)

            input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
            input_var = Variable(torch.FloatTensor(input_np))
            output = model(input_var)

            k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

            pytorch_output = output.data.numpy()
            keras_output = k_model.predict(input_np)

            error = np.max(pytorch_output - keras_output)
            print(error)
            if max_error < error:
                max_error = error

        print('Max error: {0}'.format(max_error))

if __name__ == '__main__':
    unittest.main()

##>>>>>>>>>>>>>./tests/layers/concat_many.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestConcatMany(nn.Module):
    """Module for Concatenation (2 or many layers) testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestConcatMany, self).__init__()
        self.conv2_1 = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)
        self.conv2_2 = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)
        self.conv2_3 = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)

    def forward(self, x):
        x = torch.cat([
            self.conv2_1(x),
            self.conv2_2(x),
            self.conv2_3(x)
        ], dim=1)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 100)
        out = np.random.randint(1, 100)

        model = TestConcatMany(inp, out, kernel_size, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/softmax.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestSoftmax(nn.Module):
    """Module for Softmax activation conversion testing
    """

    def __init__(self, inp=10, out=16, bias=True):
        super(TestSoftmax, self).__init__()
        self.linear = nn.Linear(inp, out, bias=True)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        inp = np.random.randint(1, 100)
        out = np.random.randint(1, 100)
        model = TestSoftmax(inp, out, inp % 2)

        input_np = np.random.uniform(-1.0, 1.0, (1, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp,), verbose=True, change_ordering=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/const.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestConst(nn.Module):
    """Module for Const conversion testing
    """

    def __init__(self, inp=10, out=16, bias=True):
        super(TestConst, self).__init__()
        self.linear = nn.Linear(inp, out, bias=False)

    def forward(self, x):
        x = self.linear(x) * 2.0
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        inp = np.random.randint(1, 100)
        out = np.random.randint(1, 100)
        model = TestConst(inp, out, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp))
        input_var = Variable(torch.FloatTensor(input_np))

        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/layers/lrelu.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestLeakyRelu(nn.Module):
    """Module for LeakyReLu conversion testing
    """

    def __init__(self, inp=10, out=16, bias=True):
        super(TestLeakyRelu, self).__init__()
        self.linear = nn.Linear(inp, out, bias=True)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        inp = np.random.randint(1, 100)
        out = np.random.randint(1, 100)
        model = TestLeakyRelu(inp, out, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/__init__.py======================================



##>>>>>>>>>>>>>./tests/models/menet.py======================================


import numpy as np
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras

"""
    MENet, implemented in PyTorch.
    Original paper: 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# 0.034489512



def depthwise_conv3x3(channels,
                      stride):
    return nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=channels,
        bias=False)


def group_conv1x1(in_channels,
                  out_channels,
                  groups):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        groups=groups,
        bias=False)

def channel_shuffle(x,
                    groups):
    """Channel Shuffle operation from ShuffleNet [arxiv: 1707.01083]
    Arguments:
        x (Tensor): tensor to shuffle.
        groups (int): groups to be split
    """
    batch, channels, height, width = x.size()
    #assert (channels % groups == 0)
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x


class ChannelShuffle(nn.Module):

    def __init__(self,
                 channels,
                 groups):
        super(ChannelShuffle, self).__init__()
        #assert (channels % groups == 0)
        if channels % groups != 0:
            raise ValueError('channels must be divisible by groups')
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)

class ShuffleInitBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(ShuffleInitBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activ = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x



def conv1x1(in_channels,
            out_channels):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        bias=False)


def conv3x3(in_channels,
            out_channels,
            stride):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class MEModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 side_channels,
                 groups,
                 downsample,
                 ignore_group):
        super(MEModule, self).__init__()
        self.downsample = downsample
        mid_channels = out_channels // 4

        if downsample:
            out_channels -= in_channels

        # residual branch
        self.compress_conv1 = group_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            groups=(1 if ignore_group else groups))
        self.compress_bn1 = nn.BatchNorm2d(num_features=mid_channels)
        self.c_shuffle = ChannelShuffle(
            channels=mid_channels,
            groups=(1 if ignore_group else groups))
        self.dw_conv2 = depthwise_conv3x3(
            channels=mid_channels,
            stride=(2 if self.downsample else 1))
        self.dw_bn2 = nn.BatchNorm2d(num_features=mid_channels)
        self.expand_conv3 = group_conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            groups=groups)
        self.expand_bn3 = nn.BatchNorm2d(num_features=out_channels)
        if downsample:
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.activ = nn.ReLU(inplace=True)

        # fusion branch
        self.s_merge_conv = conv1x1(
            in_channels=mid_channels,
            out_channels=side_channels)
        self.s_merge_bn = nn.BatchNorm2d(num_features=side_channels)
        self.s_conv = conv3x3(
            in_channels=side_channels,
            out_channels=side_channels,
            stride=(2 if self.downsample else 1))
        self.s_conv_bn = nn.BatchNorm2d(num_features=side_channels)
        self.s_evolve_conv = conv1x1(
            in_channels=side_channels,
            out_channels=mid_channels)
        self.s_evolve_bn = nn.BatchNorm2d(num_features=mid_channels)

    def forward(self, x):
        identity = x
        # pointwise group convolution 1
        x = self.activ(self.compress_bn1(self.compress_conv1(x)))
        x = self.c_shuffle(x)
        # merging
        y = self.s_merge_conv(x)
        y = self.s_merge_bn(y)
        y = self.activ(y)
        # depthwise convolution (bottleneck)
        x = self.dw_bn2(self.dw_conv2(x))
        # evolution
        y = self.s_conv(y)
        y = self.s_conv_bn(y)
        y = self.activ(y)
        y = self.s_evolve_conv(y)
        y = self.s_evolve_bn(y)
        y = F.sigmoid(y)
        x = x * y
        # pointwise group convolution 2
        x = self.expand_bn3(self.expand_conv3(x))
        # identity branch
        if self.downsample:
            identity = self.avgpool(identity)
            x = torch.cat((x, identity), dim=1)
        else:
            x = x + identity
        x = self.activ(x)
        return x


class MENet(nn.Module):

    def __init__(self,
                 block_channels,
                 side_channels,
                 groups,
                 num_classes=1000):
        super(MENet, self).__init__()
        input_channels = 3
        block_layers = [4, 8, 4]

        self.features = nn.Sequential()
        self.features.add_module("init_block", ShuffleInitBlock(
            in_channels=input_channels,
            out_channels=block_channels[0]))

        for i in range(len(block_channels) - 1):
            stage = nn.Sequential()
            in_channels_i = block_channels[i]
            out_channels_i = block_channels[i + 1]
            for j in range(block_layers[i]):
                stage.add_module("unit_{}".format(j + 1), MEModule(
                    in_channels=(in_channels_i if j == 0 else out_channels_i),
                    out_channels=out_channels_i,
                    side_channels=side_channels,
                    groups=groups,
                    downsample=(j == 0),
                    ignore_group=(i == 0 and j == 0)))
            self.features.add_module("stage_{}".format(i + 1), stage)

        self.features.add_module('final_pool', nn.AvgPool2d(kernel_size=7))

        self.output = nn.Linear(
            in_features=block_channels[-1],
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_menet(first_block_channels,
              side_channels,
              groups,
              pretrained=False,
              **kwargs):
    if first_block_channels == 108:
        block_channels = [12, 108, 216, 432]
    elif first_block_channels == 128:
        block_channels = [12, 128, 256, 512]
    elif first_block_channels == 160:
        block_channels = [16, 160, 320, 640]
    elif first_block_channels == 228:
        block_channels = [24, 228, 456, 912]
    elif first_block_channels == 256:
        block_channels = [24, 256, 512, 1024]
    elif first_block_channels == 348:
        block_channels = [24, 348, 696, 1392]
    elif first_block_channels == 352:
        block_channels = [24, 352, 704, 1408]
    elif first_block_channels == 456:
        block_channels = [48, 456, 912, 1824]
    else:
        raise ValueError("The {} of `first_block_channels` is not supported".format(first_block_channels))

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    net = MENet(
        block_channels=block_channels,
        side_channels=side_channels,
        groups=groups,
        **kwargs)
    return net


def menet108_8x1_g3(**kwargs):
    return get_menet(108, 8, 3, **kwargs)


def menet128_8x1_g4(**kwargs):
    return get_menet(128, 8, 4, **kwargs)


def menet160_8x1_g8(**kwargs):
    return get_menet(160, 8, 8, **kwargs)


def menet228_12x1_g3(**kwargs):
    return get_menet(228, 12, 3, **kwargs)


def menet256_12x1_g4(**kwargs):
    return get_menet(256, 12, 4, **kwargs)


def menet348_12x1_g3(**kwargs):
    return get_menet(348, 12, 3, **kwargs)


def menet352_12x1_g8(**kwargs):
    return get_menet(352, 12, 8, **kwargs)


def menet456_24x1_g3(**kwargs):
    return get_menet(456, 24, 3, **kwargs)


if __name__ == '__main__':
    max_error = 0
    for i in range(10):
        model = menet228_12x1_g3()
        for m in model.modules():
            m.training = False

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (3, 224, 224,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
#
#
# if __name__ == "__main__":
#     import numpy as np
#     import torch
#     from torch.autograd import Variable
#     net = menet228_12x1_g3(num_classes=1000)
#     input = Variable(torch.randn(1, 3, 224, 224))
#     output = net(input)
#     #print(output.size())
#     #print("net={}".format(net))
#
#     net.train()
#     net_params = filter(lambda p: p.requires_grad, net.parameters())
#     weight_count = 0
#     for param in net_params:
#         weight_count += np.prod(param.size())
#     print("weight_count={}".format(weight_count))
#

##>>>>>>>>>>>>>./tests/models/squeezenet.py======================================


# flake8: noqa

import keras  # work around segfault
import sys
import numpy as np
import math

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

from pytorch2keras.converter import pytorch_to_keras

# The code from torchvision
import math
import torch
import torch.nn as nn
import torch.nn.init as init


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


if __name__ == '__main__':
    max_error = 0
    for i in range(10):
        model = SqueezeNet(version=1.1)
        for m in model.modules():
            m.training = False

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (3, 224, 224,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/models/resnet18_channels_last.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision


def check_error(output, k_model, input_np, epsilon=1e-5):
    pytorch_output = output.data.numpy()
    keras_output = k_model.predict(input_np)

    error = np.max(pytorch_output - keras_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        model = torchvision.models.resnet18()
        model.eval()

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (3, 224, 224,), verbose=True,  change_ordering=True)

        error = check_error(output, k_model, input_np.transpose(0, 2, 3, 1))
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/models/senet.py======================================


import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.models import ResNet
from pytorch2keras.converter import pytorch_to_keras


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet18(num_classes):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet101(num_classes):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


class CifarSEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class CifarSEResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEResNet, self).__init__()
        self.inplane = 16
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CifarSEPreActResNet(CifarSEResNet):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEPreActResNet, self).__init__(block, n_size, num_classes, reduction)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


if __name__ == '__main__':
    max_error = 0
    for i in range(10):
        model = CifarSEResNet(CifarSEBasicBlock, 3)
        for m in model.modules():
            m.training = False

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (3, 224, 224,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/models/vgg11.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision


def check_error(output, k_model, input_np, epsilon=1e-5):
    pytorch_output = output.data.numpy()
    keras_output = k_model.predict(input_np)

    error = np.max(pytorch_output - keras_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        model = torchvision.models.vgg11_bn()
        model.eval()

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (3, 224, 224,), verbose=True)

        error = check_error(output, k_model, input_np)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
##>>>>>>>>>>>>>./tests/models/resnet18.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision


def check_error(output, k_model, input_np, epsilon=1e-5):
    pytorch_output = output.data.numpy()
    keras_output = k_model.predict(input_np)

    error = np.max(pytorch_output - keras_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        model = torchvision.models.resnet18()
        model.eval()

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (3, 224, 224,), verbose=True)

        error = check_error(output, k_model, input_np)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
##>>>>>>>>>>>>>./tests/models/__init__.py======================================



##>>>>>>>>>>>>>./tests/models/mobilinet.py======================================


import numpy as np
import torch
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torch.nn as nn
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(input_size//32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )

        # self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    max_error = 0
    for i in range(10):
        model = MobileNetV2()
        for m in model.modules():
            m.training = False

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (3, 224, 224,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/models/preresnet18.py======================================


"""
Model from https://github.com/osmr/imgclsmob/tree/master/pytorch/models
"""

import numpy as np
import torch
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision

import os
import torch.nn as nn
import torch.nn.init as init


class PreResConv(nn.Module):
    """
    PreResNet specific convolution block, with pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(PreResConv, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        x_pre_activ = x
        x = self.conv(x)
        return x, x_pre_activ


def conv1x1(in_channels,
            out_channels,
            stride):
    """
    Convolution 1x1 layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False)


def preres_conv1x1(in_channels,
                   out_channels,
                   stride):
    """
    1x1 version of the PreResNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    return PreResConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0)


def preres_conv3x3(in_channels,
                   out_channels,
                   stride):
    """
    3x3 version of the PreResNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    return PreResConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1)


class PreResBlock(nn.Module):
    """
    Simple PreResNet block for residual path in ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(PreResBlock, self).__init__()
        self.conv1 = preres_conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.conv2 = preres_conv3x3(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=1)

    def forward(self, x):
        x, x_pre_activ = self.conv1(x)
        x, _ = self.conv2(x)
        return x, x_pre_activ


class PreResBottleneck(nn.Module):
    """
    PreResNet bottleneck block for residual path in PreResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer of the block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 conv1_stride):
        super(PreResBottleneck, self).__init__()
        mid_channels = out_channels // 4

        self.conv1 = preres_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=(stride if conv1_stride else 1))
        self.conv2 = preres_conv3x3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=(1 if conv1_stride else stride))
        self.conv3 = preres_conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=1)

    def forward(self, x):
        x, x_pre_activ = self.conv1(x)
        x, _ = self.conv2(x)
        x, _ = self.conv3(x)
        return x, x_pre_activ


class PreResUnit(nn.Module):
    """
    PreResNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer of the block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck,
                 conv1_stride):
        super(PreResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if bottleneck:
            self.body = PreResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                conv1_stride=conv1_stride)
        else:
            self.body = PreResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
        if self.resize_identity:
            self.identity_conv = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)

    def forward(self, x):
        identity = x
        x, x_pre_activ = self.body(x)
        if self.resize_identity:
            identity = self.identity_conv(x_pre_activ)
        x = x + identity
        return x


class PreResInitBlock(nn.Module):
    """
    PreResNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(PreResInitBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activ = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


class PreResActivation(nn.Module):
    """
    PreResNet pure pre-activation block without convolution layer. It's used by itself as the final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(PreResActivation, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class PreResNet(nn.Module):
    """
    PreResNet model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    in_channels : int, default 3
        Number of input channels.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 conv1_stride,
                 in_channels=3,
                 num_classes=1000):
        super(PreResNet, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module("init_block", PreResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 1 if (i == 0) or (j != 0) else 2
                stage.add_module("unit{}".format(j + 1), PreResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck,
                    conv1_stride=conv1_stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('post_activ', PreResActivation(in_channels=in_channels))
        self.features.add_module('final_pool', nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_preresnet(blocks,
                  conv1_stride=True,
                  width_scale=1.0,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join('~', '.torch', 'models'),
                  **kwargs):
    """
    Create PreResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14:
        layers = [2, 2, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)

    net = PreResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        import torch
        from .model_store import get_model_file
        net.load_state_dict(torch.load(get_model_file(
            model_name=model_name,
            local_model_store_dir_path=root)))

    return net


def preresnet10(**kwargs):
    """
    PreResNet-10 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=10, model_name="preresnet10", **kwargs)


def preresnet12(**kwargs):
    """
    PreResNet-12 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=12, model_name="preresnet12", **kwargs)


def preresnet14(**kwargs):
    """
    PreResNet-14 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=14, model_name="preresnet14", **kwargs)


def preresnet16(**kwargs):
    """
    PreResNet-16 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=16, model_name="preresnet16", **kwargs)


def preresnet18_wd4(**kwargs):
    """
    PreResNet-18 model with 0.25 width scale from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=18, width_scale=0.25, model_name="preresnet18_wd4", **kwargs)


def preresnet18_wd2(**kwargs):
    """
    PreResNet-18 model with 0.5 width scale from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=18, width_scale=0.5, model_name="preresnet18_wd2", **kwargs)


def preresnet18_w3d4(**kwargs):
    """
    PreResNet-18 model with 0.75 width scale from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=18, width_scale=0.75, model_name="preresnet18_w3d4", **kwargs)


def preresnet18(**kwargs):
    """
    PreResNet-18 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=18, model_name="preresnet18", **kwargs)


def preresnet34(**kwargs):
    """
    PreResNet-34 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=34, model_name="preresnet34", **kwargs)


def preresnet50(**kwargs):
    """
    PreResNet-50 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=50, model_name="preresnet50", **kwargs)


def preresnet50b(**kwargs):
    """
    PreResNet-50 model with stride at the second convolution in bottleneck block from 'Identity Mappings in Deep
    Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=50, conv1_stride=False, model_name="preresnet50b", **kwargs)


def preresnet101(**kwargs):
    """
    PreResNet-101 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=101, model_name="preresnet101", **kwargs)


def preresnet101b(**kwargs):
    """
    PreResNet-101 model with stride at the second convolution in bottleneck block from 'Identity Mappings in Deep
    Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=101, conv1_stride=False, model_name="preresnet101b", **kwargs)


def preresnet152(**kwargs):
    """
    PreResNet-152 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=152, model_name="preresnet152", **kwargs)


def preresnet152b(**kwargs):
    """
    PreResNet-152 model with stride at the second convolution in bottleneck block from 'Identity Mappings in Deep
    Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=152, conv1_stride=False, model_name="preresnet152b", **kwargs)


def preresnet200(**kwargs):
    """
    PreResNet-200 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=200, model_name="preresnet200", **kwargs)


def preresnet200b(**kwargs):
    """
    PreResNet-200 model with stride at the second convolution in bottleneck block from 'Identity Mappings in Deep
    Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=200, conv1_stride=False, model_name="preresnet200b", **kwargs)


if __name__ == '__main__':
    max_error = 0
    for i in range(10):
        model = preresnet18()
        for m in model.modules():
            m.training = False

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (3, 224, 224,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/models/squeezenext.py======================================


"""
Model from https://github.com/osmr/imgclsmob/tree/master/pytorch/models
"""


import numpy as np
import torch
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision

import os
import torch.nn as nn
import torch.nn.init as init


class SqnxtConv(nn.Module):
    """
    SqueezeNext specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default (0, 0)
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=(0, 0)):
        super(SqnxtConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


class SqnxtUnit(nn.Module):
    """
    SqueezeNext unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(SqnxtUnit, self).__init__()
        if stride == 2:
            reduction_den = 1
            self.resize_identity = True
        elif in_channels > out_channels:
            reduction_den = 4
            self.resize_identity = True
        else:
            reduction_den = 2
            self.resize_identity = False

        self.conv1 = SqnxtConv(
            in_channels=in_channels,
            out_channels=(in_channels // reduction_den),
            kernel_size=1,
            stride=stride)
        self.conv2 = SqnxtConv(
            in_channels=(in_channels // reduction_den),
            out_channels=(in_channels // (2 * reduction_den)),
            kernel_size=1,
            stride=1)
        self.conv3 = SqnxtConv(
            in_channels=(in_channels // (2 * reduction_den)),
            out_channels=(in_channels // reduction_den),
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1))
        self.conv4 = SqnxtConv(
            in_channels=(in_channels // reduction_den),
            out_channels=(in_channels // reduction_den),
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0))
        self.conv5 = SqnxtConv(
            in_channels=(in_channels // reduction_den),
            out_channels=out_channels,
            kernel_size=1,
            stride=1)

        if self.resize_identity:
            self.identity_conv = SqnxtConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        identity = self.activ(identity)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x + identity
        x = self.activ(x)
        return x


class SqnxtInitBlock(nn.Module):
    """
    SqueezeNext specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(SqnxtInitBlock, self).__init__()
        self.conv = SqnxtConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=2,
            padding=1)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class SqueezeNext(nn.Module):
    """
    SqueezeNext model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    in_channels : int, default 3
        Number of input channels.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 in_channels=3,
                 num_classes=1000):
        super(SqueezeNext, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module("init_block", SqnxtInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), SqnxtUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_block', SqnxtConv(
            in_channels=in_channels,
            out_channels=final_block_channels,
            kernel_size=1,
            stride=1))
        in_channels = final_block_channels
        self.features.add_module('final_pool', nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_squeezenext(version,
                    width_scale,
                    model_name=None,
                    pretrained=False,
                    root=os.path.join('~', '.torch', 'models'),
                    **kwargs):
    """
    Create SqueezeNext model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of SqueezeNet ('23' or '23v5').
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    init_block_channels = 64
    final_block_channels = 128
    channels_per_layers = [32, 64, 128, 256]

    if version == '23':
        layers = [6, 6, 8, 1]
    elif version == '23v5':
        layers = [2, 4, 14, 1]
    else:
        raise ValueError("Unsupported SqueezeNet version {}".format(version))

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)
        final_block_channels = int(final_block_channels * width_scale)

    net = SqueezeNext(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        import torch
        from .model_store import get_model_file
        net.load_state_dict(torch.load(get_model_file(
            model_name=model_name,
            local_model_store_dir_path=root)))

    return net


def sqnxt23_w1(**kwargs):
    """
    1.0-SqNxt-23 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_squeezenext(version="23", width_scale=1.0, model_name="sqnxt23_w1", **kwargs)


def sqnxt23_w3d2(**kwargs):
    """
    0.75-SqNxt-23 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_squeezenext(version="23", width_scale=1.5, model_name="sqnxt23_w3d2", **kwargs)


def sqnxt23_w2(**kwargs):
    """
    0.5-SqNxt-23 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_squeezenext(version="23", width_scale=2.0, model_name="sqnxt23_w2", **kwargs)


def sqnxt23v5_w1(**kwargs):
    """
    1.0-SqNxt-23v5 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_squeezenext(version="23v5", width_scale=1.0, model_name="sqnxt23v5_w1", **kwargs)


def sqnxt23v5_w3d2(**kwargs):
    """
    0.75-SqNxt-23v5 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_squeezenext(version="23v5", width_scale=1.5, model_name="sqnxt23v5_w3d2", **kwargs)


def sqnxt23v5_w2(**kwargs):
    """
    0.5-SqNxt-23v5 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_squeezenext(version="23v5", width_scale=2.0, model_name="sqnxt23v5_w2", **kwargs)




if __name__ == '__main__':
    max_error = 0
    for i in range(10):
        model = sqnxt23_w1()
        for m in model.modules():
            m.training = False

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (3, 224, 224,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))

##>>>>>>>>>>>>>./tests/models/resnet34.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision


def check_error(output, k_model, input_np, epsilon=1e-5):
    pytorch_output = output.data.numpy()
    keras_output = k_model.predict(input_np)

    error = np.max(pytorch_output - keras_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        model = torchvision.models.resnet34()
        model.eval()

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (3, 224, 224,), verbose=True)

        error = check_error(output, k_model, input_np)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
##>>>>>>>>>>>>>./tests/models/alexnet.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision


def check_error(output, k_model, input_np, epsilon=1e-5):
    pytorch_output = output.data.numpy()
    keras_output = k_model.predict(input_np)

    error = np.max(pytorch_output - keras_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        model = torchvision.models.AlexNet()
        model.eval()

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (3, 224, 224,), verbose=True)

        error = check_error(output, k_model, input_np)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
##>>>>>>>>>>>>>./tests/models/resnet50.py======================================


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision


def check_error(output, k_model, input_np, epsilon=1e-5):
    pytorch_output = output.data.numpy()
    keras_output = k_model.predict(input_np)

    error = np.max(pytorch_output - keras_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        model = torchvision.models.resnet50()
        model.eval()

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (3, 224, 224,), verbose=True)

        error = check_error(output, k_model, input_np)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
##>>>>>>>>>>>>>./setup.py======================================


from setuptools import setup, find_packages


try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements


# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt', session='null')


# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]


with open('README.md') as f:
  long_description = f.read()


setup(name='pytorch2keras',
      version='0.1.4',
      description='The deep learning models convertor',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/nerox8664/pytorch2keras',
      author='Grigory Malivenko',
      author_email='nerox8664@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=reqs,
      zip_safe=False)
