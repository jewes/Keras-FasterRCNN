from keras import Model
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input
from keras.layers import TimeDistributed

from keras_frcnn.RoiPoolingConv import RoiPoolingConv


def get_weight_path():
    return None

def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length/16

    return get_output_length(width), get_output_length(height)


def nn_base(input_tensor, trainable=True):
    # block 1
    x = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', name='block1_conv1')(input_tensor)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # block 2
    x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', name='block2_conv1')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # block 3
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', name='block3_conv1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', name='block3_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    return x


def rpn(base_layers, num_anchors):
    x = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='uniform', name='rpn_out_regr')(x)

    return [x_class, x_regr]


def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=True):
    pooling_regions = 7
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'), name='td1')(out_roi_pool)
    out = TimeDistributed(Dense(128, activation='relu', name='fc1'), name='td2')(out)
    out = TimeDistributed(Dense(128, activation='relu', name='fc2'), name='td3')(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]


if __name__ == '__main__':
    input_shape_img = (None, None, 3)

    # input placeholder 정의
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    # base network(feature extractor) 정의 (resnet, VGG, Inception, Inception Resnet V2, etc)
    shared_layers = nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    # RPN 정의
    num_anchors = 9
    rpn = rpn(shared_layers, num_anchors)

    # detection network 정의
    classifier = classifier(shared_layers, roi_input, 300, nb_classes=2, trainable=True)

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all: Model = Model([img_input, roi_input], rpn[:2] + classifier)
    print(model_all.summary(positions=[.3, .6, .8, 1.]))
