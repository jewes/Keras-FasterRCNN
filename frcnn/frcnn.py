from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Model


class FRCNN(object):

    def __init__(self):
        self.sub_sample = 16  # 2^4 = 16, there are 4 max pooling layers

    def _build_backbone_network(self, img_input):
        # block 1
        x = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', name='block1_conv1')(img_input)
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

    def _rpn(self, base_layers, num_anchors):
        x = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', name='rpn_conv1')(base_layers)

        x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
        x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='uniform', name='rpn_out_regr')(x)

        return [x_class, x_regr]

    def build_rpn_model(self, num_anchor):
        img_input = Input(shape=(None, None, 3))
        backbone_net = self._build_backbone_network(img_input)
        rpn_output = self._rpn(backbone_net, num_anchor)

        return Model(inputs=img_input, outputs=rpn_output)


if __name__ == '__main__':
    frcnn = FRCNN()
    rpn_model = frcnn.build_rpn_model(9)
    print(rpn_model.summary(positions=[.3, .6, .8, 1.]))
