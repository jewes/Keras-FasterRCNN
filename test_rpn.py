from keras.models import load_model
from frcnn.datagen import process_image


if __name__ == '__main__':
    model_rpn = load_model('rpn_model.h5', compile=False)
    img_x, w, h, rw, rh = process_image('dataset/images/car.jpg')
    p_rpn_cls, p_rpn_regr = model_rpn.predict_on_batch(img_x)

    # todo run nms to filter the output
    pass
