import numpy as np
from keras import objectives
from keras import backend as K
import tensorflow as tf

_EPSILON = K.epsilon()


def iou_loss(target, output):
    """Compute IoU between an output tensor (prediction) and a target tensor.
    # Arguments
        target: A label/grount truth tensor with the same shape as `output`.
        output: A prediction/logits tensor representing.
    # Returns
        A tensor.
    """

    assert target.shape == output.shape
    pred = tf.reshape(output, [-1])
    gt_labels = tf.reshape(target, [-1])

    mul_x_y = tf.multiply(pred, gt_labels)
    inter = tf.reduce_sum(mul_x_y)

    union = tf.reduce_sum(tf.subtract(tf.add(pred, gt_labels), mul_x_y))

    iou = tf.divide(inter, union)
    return tf.subtract(tf.constant(1.0, dtype=tf.float32), iou)


def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)


def _loss_np(y_true, y_pred):
    y_pred = np.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return np.mean(out, axis=-1)


def test_iou_loss():
    shape = (5, 2,3)
    y_a = np.ones(shape)
    y_b = np.zeros(shape)
    out1 = K.eval(iou_loss(K.variable(y_a), K.variable(y_b)))
    assert out1 == 1.0

    y_a = np.ones(shape)
    y_b = np.ones(shape)
    out1 = K.eval(iou_loss(K.variable(y_a), K.variable(y_b)))
    assert out1 == 0.0

    y_a = np.random.randint(0, 2, shape)
    y_b = np.random.randint(0, 2, shape)
    out1 = K.eval(iou_loss(K.variable(y_a), K.variable(y_b)))
    assert out1 == 0.0

def check_loss(_shape):
    if _shape == '2d':
        shape = (6, 7)
    elif _shape == '3d':
        shape = (5, 6, 7)
    elif _shape == '4d':
        shape = (8, 5, 6, 7)
    elif _shape == '5d':
        shape = (9, 8, 5, 6, 7)

    y_a = np.random.random(shape)
    y_b = np.random.random(shape)

    # out1 = K.eval(_loss_tensor(K.variable(y_a), K.variable(y_b)))
    out1 = K.eval(iou_loss(K.variable(y_a), K.variable(y_b)))

    out2 = _loss_np(y_a, y_b)

    assert out1.shape == out2.shape
    assert out1.shape == shape[:-1]
    print(np.linalg.norm(out1))
    print(np.linalg.norm(out2))
    print(np.linalg.norm(out1-out2))


def test_loss():
    shape_list = ['2d', '3d', '4d', '5d']
    for _shape in shape_list:
        check_loss(_shape)
        print('======================')


if __name__ == '__main__':
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
    with tf.device(DEVICE):
        #test_loss()
        test_iou_loss()
