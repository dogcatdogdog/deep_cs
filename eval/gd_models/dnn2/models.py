from tensorflow.keras.layers import Input, BatchNormalization, Add, Dense, Activation, Multiply, Lambda
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from .graph_layers import GraphConvolution

def conv_block_residual(input, support, filters, supports, inputXmask, conv_shortcut=False, activation="relu"):
    features = input
    reg=None
    for i in range(len(filters)):
        filter_i = filters[i]
        x = GraphConvolution(units=filter_i, support=support, activation=activation, kernel_regularizer=reg)([features, supports])
        x = BatchNormalization()(x)
        features = x
    
    shortcut = None
    if(conv_shortcut):
        shortcut = GraphConvolution(units=filters[-1], support=support, activation=activation, kernel_regularizer=reg)([input, supports])
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = input
    x = Add()([x, shortcut])
    x = Activation(activation)(x)
    
    # [FIX 1]: Use Multiply() layer instead of Python '*' operator to preserve Keras graph history
    x = Multiply()([x, inputXmask]) 
    return x

def build_model(Nmax, initial_features_vector_size, support_size, loss_is_tsnet=False,use_DC=False,use_TH=False):
    if(Nmax == -1):
        Nmax = None
    inputDM = Input(shape=(Nmax, Nmax), name="input_DM")  
    inputS = Input(shape=(support_size, Nmax, Nmax), name="input_supports") 
    inputX = Input(shape=(Nmax, initial_features_vector_size), name="input_features") 
    inputXmask = Input(shape=(Nmax,1), name="input_nodesMask") 
    
    inputs = [inputDM, inputXmask, inputX, inputS]
    
    if(loss_is_tsnet):
        inputSigma = Input(shape=(Nmax,1), name="input_sigma") 
        inputs.append(inputSigma)
    if(use_DC):
        inputDC = Input(shape=(Nmax, 1), name="input_DC")
        inputs.append(inputDC)
    if(use_TH):
        inputTH = Input(shape=(Nmax, Nmax), name="input_TH")
        inputs.append(inputTH)
    
    last_support = min(2, support_size)
    
    # [FIX 1]: Use Multiply() layer instead of Python '*' operator
    x = Multiply()([inputX, inputXmask]) 
    
    # Features extraction
    x = conv_block_residual(x, last_support, [16, 16, 32], inputS, inputXmask, conv_shortcut=True)
    x = conv_block_residual(x, support_size, [16, 16, 32], inputS, inputXmask)
    stack1 = conv_block_residual(x, support_size, [16, 16, 32], inputS, inputXmask)

    x = conv_block_residual(stack1, last_support, [32, 32, 64], inputS, inputXmask, conv_shortcut=True)
    x = conv_block_residual(x, support_size, [32, 32, 64], inputS, inputXmask)
    x = conv_block_residual(x, support_size, [32, 32, 64], inputS, inputXmask)
    stack2 = conv_block_residual(x, support_size, [32, 32, 64], inputS, inputXmask)

    x = conv_block_residual(stack2, last_support, [64, 64, 128], inputS, inputXmask, conv_shortcut=True)
    x = conv_block_residual(x, support_size, [64, 64, 128], inputS, inputXmask)
    x = conv_block_residual(x, support_size, [64, 64, 128], inputS, inputXmask)
    x = conv_block_residual(x, support_size, [64, 64, 128], inputS, inputXmask)
    x = conv_block_residual(x, support_size, [64, 64, 128], inputS, inputXmask)
    stack3 = conv_block_residual(x, support_size, [64, 64, 128], inputS, inputXmask)

    x = conv_block_residual(stack3, last_support, [128, 128, 128], inputS, inputXmask, conv_shortcut=True)
    x = conv_block_residual(x, last_support, [128, 128, 128], inputS, inputXmask)
    stack4 = conv_block_residual(x, last_support, [128, 128, 128], inputS, inputXmask)

    # Regression
    x = stack4
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(2, name="output")(x)
    
    # [FIX 2]: Safely bind unused inputs to the graph to prevent TF2 Disconnected Graph error.
    # We multiply them by 0.0 so they don't affect the coordinates at all.
    used_inputs = [inputX, inputXmask, inputS]
    unused_inputs = [inp for inp in inputs if inp not in used_inputs]
    
    if len(unused_inputs) > 0:
        def sink_unused(args):
            main_tensor = args[0]
            for dummy in args[1:]:
                main_tensor = main_tensor + 0.0 * K.sum(dummy)
            return main_tensor
        
        x = Lambda(sink_unused)([x] + unused_inputs)
    
    return Model(inputs, x)

if __name__ == "__main__":
    Nmax = 128
    initial_features_vector_size = 2 
    support_size = 4 
    model = build_model(Nmax, initial_features_vector_size, support_size, loss_is_tsnet=False)
    model.summary()