
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Add
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.layers import Layer

def head(endpoints, embedding_dim, is_training):
    endpoints['emb1'] = GlobalAveragePooling2D(data_format='channels_last')(endpoints['Conv2d_1_pointwise'])
    endpoints['fc_layer1_1'] = Dense(1024, activation='relu', kernel_initializer='Orthogonal')(endpoints['emb1'])
    endpoints['bn_1_1'] = BN()(endpoints['fc_layer1_1'])
    endpoints['feature1'] = Dense(embedding_dim, activation=None, kernel_initializer='Orthogonal')(endpoints['bn_1_1'])

    endpoints['emb2'] = GlobalAveragePooling2D(data_format='channels_last')(endpoints['Conv2d_5_pointwise'])
    endpoints['fc_layer2_1'] = Dense(1024, activation='relu', kernel_initializer='Orthogonal')(endpoints['emb2'])
    endpoints['bn_2_1'] = BN()(endpoints['fc_layer2_1'])
    endpoints['feature2'] = Dense(embedding_dim, activation=None, kernel_initializer='Orthogonal')(endpoints['bn_2_1'])

    endpoints['emb3'] = GlobalAveragePooling2D(data_format='channels_last')(endpoints['Conv2d_13_pointwise'])
    endpoints['fc_layer3_1'] = Dense(1024, activation='relu', kernel_initializer='Orthogonal')(endpoints['emb3'])
    endpoints['bn_3_1'] = BN()(endpoints['fc_layer3_1'])
    endpoints['feature3'] = Dense(embedding_dim, activation=None, kernel_initializer='Orthogonal')(endpoints['bn_3_1'])

    # endpoints['emb4'] = GlobalAveragePooling2D(data_format='channels_last')(endpoints['Conv2d_11_pointwise'])
    # endpoints['fc_layer4_1'] = Dense(1024, activation='relu', kernel_initializer='Orthogonal')(endpoints['emb4'])
    # endpoints['bn_4_1'] = BN()(endpoints['fc_layer4_1'])
    # endpoints['feature4'] = Dense(embedding_dim, activation=None, kernel_initializer='Orthogonal')(endpoints['bn_4_1'])
    #
    # endpoints['emb5'] = GlobalAveragePooling2D(data_format='channels_last')(endpoints['Conv2d_13_pointwise'])
    # endpoints['fc_layer5_1'] = Dense(1024, activation='relu', kernel_initializer='Orthogonal')(endpoints['emb5'])
    # endpoints['bn_5_1'] = BN()(endpoints['fc_layer5_1'])
    # endpoints['feature5'] = Dense(embedding_dim, activation=None, kernel_initializer='Orthogonal')(endpoints['bn_5_1'])

    # endpoints['fusion_layer'] = FusionLayer()([endpoints['feature1'], endpoints['feature2'], endpoints['feature3'],
    #                                            endpoints['feature4'],endpoints['feature5']])
    endpoints['fusion_layer'] = FusionLayer()([endpoints['feature1'], endpoints['feature2'], endpoints['feature3']])

    endpoints['emb'] = endpoints['emb_raw'] = endpoints['fusion_layer']
    return endpoints


class FusionLayer(Layer):
    def __init__(self,**kwargs):
        super(FusionLayer,self).__init__(**kwargs)

    def build(self, input_shape):
        # create trainable weights
        self.a = self.add_weight(name='a',
                                  shape=(1,),
                                  initializer = 'uniform',
                                  trainable=True)
        self.b = self.add_weight(name='b',
                                  shape=(1,),
                                  initializer='uniform',
                                  trainable=True)
        self.c = self.add_weight(name='c',
                                  shape=(1,),
                                  initializer='uniform',
                                  trainable=True)
        # self.d = self.add_weight(name='d',
        #                           shape=(1,),
        #                           initializer='uniform',
        #                           trainable=True)
        # self.e = self.add_weight(name='e',
        #                          shape=(1,),
        #                          initializer='uniform',
        #                          trainable=True)
        super(FusionLayer,self).build(input_shape)


    def call(self,x):
        # A,B,C,D,E = x
        # result = Add()([self.a * A, self.b * B, self.c * C, self.d * D,self.e * E])
        A, B, C= x
        result = Add()([self.a * A, self.b * B, self.c * C])
        return result


    def compute_output_shape(self, input_shape):
        return input_shape[0]






