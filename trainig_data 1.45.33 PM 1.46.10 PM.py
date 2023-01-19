
import glob
import cv2
import numpy as np
import os
from tensorflow.keras.applications.vgg16 import VGG16
import seaborn as sns
import matplotlib.pyplot as plt
print(os.listdir("dataset/"))

SIZE = 256

train_images=[]
train_labels=[]

for directory_path in glob.glob("dataset/train/*"):
    labels=directory_path.split("\\")[-1]
    print(labels)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img=cv2.imread(img_path, cv2.IMREAD_COLOR)
        img=cv2.resize(img, (SIZE, SIZE))
        img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(labels)

train_images=np.array(train_images)
train_labels=np.array(train_labels)

test_images=[]
test_labels=[]
for directory_path in glob.glob("dataset/test/*"):
    fruit_label=directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)
test_images=np.array(test_images)
test_labels=np.array(test_labels)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded=le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded=le.transform(train_labels)
x_train,y_train,x_test,y_test=train_images,train_labels_encoded,test_images,test_labels_encoded

x_train, x_test=x_train / 255.0, x_test / 255.0

VGG_Model=VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE))
for layer in VGG_Model.layers:
    layer.trainable = False
VGG_Model.summary()

feature_extractor=VGG_Model.predict(x_train)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)
X_for_training = features

print(X_for_training)
import xgboost as xgb
import pickle
model = xgb.XGBClassifier()
filename = 'finalized_model.sav'
history=model.fit(X_for_training, y_train)
pickle.dump(history, open(filename, 'wb'))

# model.save('alg/model.h5')
X_test_feature = VGG_Model.predict(x_test)
X_test_features=X_test_feature.reshape(X_test_feature.shape[0], -1)

prediction = model.predict(X_test_features)

prediction = le.inverse_transform((prediction))

from sklearn import metrics
print("Accuracy = ", metrics.accuracy_score(test_labels, prediction))

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(test_labels, prediction)
# sns.heatmap(cm,annot=True)

n=np.random.randint(0,x_test.shape[0])
img=x_test[n]
print(img)
plt.imshow(img)
# cv2.imshow(img)
# plt.savefig(r"Y:\VGG_with_XGBoost\abc.jpg")
input_img=np.expand_dims(img, axis=0)
input_img_feature=VGG_Model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0],-1)
prediction=model.predict(input_img_features)
prediction=le.inverse_transform(prediction)
print('prediction image is:', prediction)
print("actual image is: ", test_labels[n])


