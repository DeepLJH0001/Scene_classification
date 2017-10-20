#coding=utf-8
from keras.models import Model
from keras.layers import Input,GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.preprocessing.image import load_img,img_to_array
import pandas as pd
import numpy as np
import os
def get_model(type = 0):
    if type==0:
        input_x = Input((224,224,3))#tf
        from keras.applications.resnet50 import ResNet50,preprocess_input
        model = ResNet50(input_tensor=input_x,weights='imagenet',include_top=False)
        func = preprocess_input

    elif type == 1:
        input_x = Input((299,299,3))#tf
        from keras.applications.inception_v3 import InceptionV3,preprocess_input
        model = InceptionV3(input_tensor=input_x,weights='imagenet',include_top=False)
        func = preprocess_input

    elif type == 2:
        input_x = Input((299,299,3))#tf
        from keras.applications.xception import Xception,preprocess_input
        model = Xception(input_tensor=input_x,weights='imagenet',include_top=False)
        func = preprocess_input

    elif type == 3:
        input_x = Input((299,299,3))#tf
        from keras.applications.mobilenet import MobileNet,preprocess_input
        model = MobileNet(input_tensor=input_x,weights='imagenet',include_top=False)
        func = preprocess_input

    elif type == 4:
        input_x = Input((299,299,3))#tf
        from keras.applications.vgg19 import VGG19,preprocess_input
        model = VGG19(input_tensor=input_x,weights='imagenet',include_top=False)
        func = preprocess_input

    return model,preprocess_input
if __name__ == '__main__':
    import json
    import h5py

    train_data_path = "./ai_challenger_scene_train_20170904/scene_train_images_20170904/"
    val_data_path = "./ai_challenger_scene_validation_20170908/scene_validation_images_20170908/"
    test_data_path = "./ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922/"

    train_data = {}
    val_data = {}
    test_data = {}
    with open("./ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json",'r') as f:
        train_data = json.load(f)
        # print(train_data)
        # exit()
    with open("./ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json",'r') as f:
        val_data = json.load(f)

    images = []
    labels = []
    for i in train_data:
        images.append(train_data_path + i['image_id'])
        labels.append(i['label_id'])
    for i in val_data:
        images.append(val_data_path + i['image_id'])
        labels.append(i['label_id'])
    import pandas as pd
    df = pd.DataFrame({'images_id':np.asarray(images),'labels':np.asarray(labels)})
    df.to_csv('./train.csv')

    base_model, preprocess_input = get_model(type=1)
    target_size = (299, 299)
    # target_size = (224,224)#resnet only#0
    # save_name = 'res50_feature.h5'#0
    save_name = 'inceptionv3_feature_ag.h5'#1
    # save_name = "Xception_feature.h5"#2
    # save_name = 'mobilenet_feature.h5'#3
    # save_name = 'vgg19_feature.h5'#4
    save_path = './feature_/'
    model = Model(input=base_model.input, output=GlobalMaxPooling2D()(base_model.output))

    BATCHSIZE = 64
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_feature = np.array([0, ])
    test_feature = np.array([0, ])
    new_labels = []
    for i in range(int(df.shape[0] / BATCHSIZE) + 1):
        imgs = []
        label = []
        for j in range(BATCHSIZE):
            imgs.append(img_to_array(load_img(df.iloc[i * BATCHSIZE + j, 0], target_size=target_size)))
            label.append(df.iloc[i*BATCHSIZE+j,1])
            if i * BATCHSIZE + j == df.shape[0] - 1:
                break
        imgs = np.asarray(imgs)
        if imgs.ndim==4:
            imgs = np.expand_dims(imgs, axis=1)
        # print(imgs.shape)
        ag_imgs = []
        for k in range(imgs.shape[0]):
            n=0
            for ag in datagen.flow(imgs[k], batch_size=1):
                ag_imgs.append(ag)
                new_labels.append(label[k])
                n+=1
                if n>5:#增强数量5张
                    break
        # for ag in datagen.flow(,batch_size=1):
        ag_imgs = np.asarray(ag_imgs)
        ag_imgs = ag_imgs.reshape(ag_imgs.shape[0],ag_imgs.shape[2],ag_imgs.shape[3],ag_imgs.shape[4])
        if i == 0:
            train_feature = model.predict(preprocess_input(ag_imgs))
        else:
            print(train_feature.shape,model.predict(preprocess_input(ag_imgs)).shape)
            train_feature = np.vstack((train_feature, model.predict(preprocess_input(ag_imgs))))
    print (train_feature.shape,len(new_labels))
    new_labels = np.asarray(new_labels)
    list_2 = [test_data_path + list for list in os.listdir(test_data_path)]
    for i in range(int(len(list_2) / BATCHSIZE) + 1):
        imgs = []
        for j in range(BATCHSIZE):
            if i * BATCHSIZE + j <len(list_2):
                imgs.append(img_to_array(load_img(list_2[i * BATCHSIZE + j], target_size=target_size)))
            if i * BATCHSIZE + j == len(list_2) - 1:
                break

        imgs = np.asarray(imgs)
        if i == 0:
            test_feature = model.predict(preprocess_input(imgs))
        else:
            test_feature = np.vstack([test_feature, model.predict(preprocess_input(imgs))])
    print (test_feature.shape, len(list_2))
    with h5py.File(save_name) as h:
        h.create_dataset("train", data=train_feature)
        h.create_dataset("test", data=test_feature)
        h.create_dataset("labels",data=new_labels)

