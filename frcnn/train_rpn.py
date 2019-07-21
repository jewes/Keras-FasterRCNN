from keras.optimizers import Adam
from .frcnn import FRCNN
from .hyper_params import H
from .datagen import TrainDataGenerator
from . import losses as losses

if __name__ == '__main__':
    num_epochs = 10

    # build model
    frcnn = FRCNN()
    model_rpn = frcnn.build_rpn_model(H.anchor_per_loc)
    model_rpn.compile(optimizer=Adam(lr=1e-5),
                      loss=[losses.rpn_loss_cls(H.num_anchor),
                            losses.rpn_loss_regr(H.num_anchor)])

    train_datagen = TrainDataGenerator("").get_train_datagen()
    for idx_epoch in range(num_epochs):
        print('Epoch {}/{}'.format(idx_epoch + 1, num_epochs))

        while True:
            X, Y = next(train_datagen)
            loss_rpn = model_rpn.train_on_batch(X, Y)
            print('Epoch {}, rpn_loss = {}'.format(idx_epoch + 1, loss_rpn))
