import numpy as np
import pandas as pd
from  scipy import ndimage
from scipy import misc
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt



train = pd.read_csv('train.csv')
test_ = pd.read_csv('test.csv')

correlations = train.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
##ticks = numpy.arange(0,9,1)
##ax.set_xticks(ticks)
##ax.set_yticks(ticks)
##ax.set_xticklabels(names)
##ax.set_yticklabels(names)
plt.show()
##plotting correlation ##


input()


img_dir = 'images/'
n_images = 1584
rows = int(467*0.2)
cols = int(526*0.2)
def encode(train, test):
    label_encoder = LabelEncoder().fit(train.species)
    labels = label_encoder.transform(train.species)
    classes = list(label_encoder.classes_)

    train = train.drop(['species', 'id'], axis=1)
    test = test_.drop('id', axis=1)

    return train, labels, test, classes

def logloss(y_true, y_pred):
    from sklearn.metrics import log_loss

    return log_loss(y_true, y_pred, eps=1e-15)


def load_image_data():
    _2d_images=[]
    img_data = np.vstack([misc.imresize(arr= ndimage.imread(img_dir+str(i+1)+'.jpg'), size= (rows, cols), interp='bilinear', mode=None).reshape(-1,)  for i in range(n_images)])

    for i in range(n_images):
        _2d_images.append(misc.imresize(arr= ndimage.imread(img_dir+str(i+1)+'.jpg'), size= (rows, cols), interp='bilinear', mode=None))

    return img_data, _2d_images

train, labels, test, classes = encode(train, test_)
train = train.values
X_test = test.values

img_data, _2d_images = load_image_data()


print("data loaded")

#### PCA #####
#pca = PCA(n_components=0.99, svd_solver='full')
#pca = PCA(n_components=0.95, svd_solver='full')
#pca = PCA(n_components=0.90, svd_solver='full')
#pca=pca.fit(train)
#train = pca.transform(train)
#X_test = pca.transform(X_test)



 
# splittrain data into train and validation
sss = StratifiedShuffleSplit(test_size=0.2, random_state=23)
for train_index, valid_index in sss.split(train, labels):
    X_train, X_valid = train[train_index], train[valid_index]
    y_train, y_valid = labels[train_index], labels[valid_index]
    X_train_img, X_valid_img = img_data[train_index], img_data[valid_index]
    
scaler = StandardScaler()
#scaler = MinMaxScaler()
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

  
print("Done")
##

## Classifying

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn import tree
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, y_train)
svm = SVC(probability=True)
svm.fit(X_train, y_train)
tree = tree.DecisionTreeClassifier()
tree.fit(X_train, y_train)


 
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


batch_size = 32
num_classes = len(classes)
epochs = 100


y_train_cat = keras.utils.to_categorical(y_train, num_classes)  #convertendo as classes em vetores binários (one hot encoding)
y_valid_cat = keras.utils.to_categorical(y_valid, num_classes)  


print("to_categorical")

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  
# monitores para o treinamento da rede
#https://keras.io/callbacks/

#Reduz a taxa de aprendizado
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  #observar a função de perda no conjunto de validação
							factor=0.2,     #fator pela qual a taxa será reduzida -> nova_taxa = taxa_atual * factor
							verbose=1, 
							patience=4,    # número de épocas, sem redução na função de perda, que o monitor espera para agir.
							min_lr=0.0001) 	#limite inferior da nova_taxa. Menor que isso e o monitor não irá reduzir mais a taxa.

#Interrompe o treinamento
early_stopping=EarlyStopping(monitor='val_loss', #observar a função de perda no conjunto de validação
							patience=4,    # número de épocas, sem redução na função de perda, que o monitor espera para agir.
							verbose=1,)    

															  #fazendo uso dos monitores criados		
history= model.fit(X_train, y_train_cat, batch_size=batch_size, epochs=epochs, callbacks=[reduce_lr, early_stopping], verbose=1, validation_data=(X_valid, y_valid_cat))




p_n = neigh.predict_proba(X_test)#, y_valid)
p_svm = svm.predict_proba(X_test)
p_tree = tree.predict_proba(X_test)
p_mlp = model.predict(X_test)


print("1-NN -  erro: ",1-neigh.score(X_valid, y_valid), " loss:", logloss(y_valid,neigh.predict_proba(X_valid)))
print("SVM -  erro: ",1-svm.score(X_valid, y_valid), " loss:", logloss(y_valid,svm.predict_proba(X_valid)))
print('Tree -  erro: ',1-tree.score(X_valid, y_valid), " loss:", logloss(y_valid,tree.predict_proba(X_valid)))

mlp_loss, mlp_acc = model.evaluate(X_valid, y_valid_cat, verbose=0)
print("MLP - erro: ", 1-mlp_acc, " loss: ", mlp_loss)



 
frame = pd.DataFrame(p_mlp,index=test_.id, columns=classes)
frame.to_csv('submisiondf_1_5.csv')




#[print('1-NN',max(p_n[i]),'SVM',max(p_svm[i]),'Tree',max(p_tree[i]),'MLP',max(p_mlp[i])) for i in range(len(p_n))]




##
##plt.plot(history.history['acc'])
##plt.plot(history.history['val_acc'])
##plt.title('model accuracy')
##plt.ylabel('accuracy')
##plt.xlabel('epoch')
##plt.legend(['train', 'val'], loc='upper left')
##plt.show()
### summarize history for loss
##plt.plot(history.history['loss'])
##plt.plot(history.history['val_loss'])
##plt.title('model loss')
##plt.ylabel('loss')
##plt.xlabel('epoch')
##plt.legend(['train', 'val'], loc='upper left')
##plt.show()
