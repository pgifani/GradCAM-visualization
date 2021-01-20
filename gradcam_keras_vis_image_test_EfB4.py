import keras
import tensorflow as tf
import vis ## keras-vis
import matplotlib.pyplot as plt
import numpy as np
import json
from keras import optimizers
import os
from vis.visualization import visualize_cam
from keras.preprocessing.image import load_img, img_to_array
from vis.utils import utils

print("keras      {}".format(keras.__version__))
print("tensorflow {}".format(tf.__version__))

from keras_efficientnets import EfficientNetB4

model_name='EfficientNetB4'   
root="./model_path/"
model_weights_path = root + model_name + "/" + "model_.05-0.63_4class_efficientnetB4.hdf5"
CLASS_INDEX = json.load(open("/_class_index_4C.json"))
paretn_root="imag_path/"
im_name="image_00015.jpg" # input frame

dest_name = 'result'

if not os.path.exists(paretn_root + '/' + dest_name ):
    os.mkdir(path + '/' + dest_name)
else:
    print('Directory:' + dest_name + ' is already exist')



num_classes=3
conv_layer_name="conv2d_128" 
s=380


def load_model():
    base_model = EfficientNetB4(input_shape=(s,s,3), weights='imagenet', include_top=False)

    #base_model.load_weights(weights_path, by_name=True)
    #base_model.summary()
    x = keras.layers.AveragePooling2D((12,12))(base_model.output)
    #x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    
    x_newfc = keras.layers.Flatten()(x)
    
    x_newfc = keras.layers.Dense(num_classes, activation='softmax', name='fc_new')(x_newfc)
    
    model = keras.models.Model(input=base_model.input, output=x_newfc)
    
    sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_weights(model_weights_path, by_name=True)

    return model



model = load_model()




model.summary()
for ilayer, layer in enumerate(model.layers):
    print("{:3.0f} {:10}".format(ilayer, layer.name))




classlabel  = []
for i_dict in range(len(CLASS_INDEX)):
    classlabel.append(CLASS_INDEX[str(i_dict)][1])
print("N of class={}".format(len(classlabel)))



impath=paretn_root + '/' + im_name 

pathm, filename = os.path.split(impath)
bpath=os.path.basename(pathm)
#_img = load_img("duck.jpg",target_size=(224,224))
_img = load_img(impath,target_size=(s,s))
#plt.imshow(_img)
#plt.show()
#rgb_img = np.expand_dims(_img, 0)
#y_pred = model.predict(rgb_img)
img               = img_to_array(_img)
#img               = preprocess_input(img)
y_pred            = model.predict(img[np.newaxis,...])
class_idxs_sorted = np.argsort(y_pred.flatten())[::-1]
topNclass         = 3
for i, idx in enumerate(class_idxs_sorted[:topNclass]):
    print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(
        i + 1,classlabel[idx],idx,y_pred[0,idx]))


# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'fc_new')
# Swap softmax with linear
model.layers[layer_idx].activation = keras.activations.linear
model = utils.apply_modifications(model)


penultimate_layer_idx = utils.find_layer_idx(model, conv_layer_name) 
class_idx  = class_idxs_sorted[0]
seed_input = img
grad_top1  = visualize_cam(model, layer_idx, class_idx, seed_input, 
                        penultimate_layer_idx = penultimate_layer_idx,#None,
                        backprop_modifier     = None,
                        grad_modifier         = None)


def plot_map(grads):
    fig = plt.figure()
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].imshow(_img)
    axes[1].imshow(_img)
    i = axes[1].imshow(grads,cmap="jet",alpha=0.8)
    fig.colorbar(i)
    plt.suptitle("Pr(class={}) = {:5.2f}".format(
                    classlabel[class_idx],
                    y_pred[0,class_idx]))
    #plt.show()
    
    

    fig.savefig(paretn_root + "/result/gragh_" + model_name + '_' + im_name)
plot_map(grad_top1)