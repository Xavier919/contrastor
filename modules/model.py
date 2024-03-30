from keras.layers import Flatten, Dense, Dropout, Flatten, Input, Conv1D, AveragePooling1D, MaxPooling1D
from keras.models import Model

def create_base_net_1D(input_shape):
    input = Input(shape=input_shape)
    x = Conv1D(32, 3, activation='relu')(input)  
    x = AveragePooling1D(pool_size=2)(x)
    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='relu')(x)
    model = Model(inputs=input, outputs=x)
    return model