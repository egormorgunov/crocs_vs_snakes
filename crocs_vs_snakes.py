import numpy as np
import matplotlib.pyplot as plt
import warnings
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import tensorflow

seed = 1842
tensorflow.random.set_seed(seed)
np.random.seed(seed)

warnings.simplefilter('ignore')

image_data = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)

train_dataset = image_data.flow_from_directory(batch_size=40,
                                               directory='Train',
                                               shuffle=True,
                                               target_size=(300, 300),
                                               subset="training",
                                               class_mode='categorical')

validation_dataset = image_data.flow_from_directory(batch_size=40,
                                                    directory='Train',
                                                    shuffle=True,
                                                    target_size=(300, 300),
                                                    subset="validation",
                                                    class_mode='categorical')

image_test_data = ImageDataGenerator(rescale=1 / 255)
test_dataset = image_test_data.flow_from_directory(
                                                   directory='Test',
                                                   shuffle=False,
                                                   target_size=(300, 300),
                                                   class_mode=None)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=[300, 300, 3]),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.3),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(2, activation='softmax')])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=3,
                                         restore_best_weights=True)
history = model.fit(train_dataset, epochs=20, validation_data=train_dataset, callbacks=callback)

loss, accuracy = model.evaluate(train_dataset)
print("Loss:", loss)
print("Accuracy:", accuracy)

prediction = model.predict(test_dataset, callbacks=callback)
predictions = []
for i in range(len(prediction)):
    a = ("%.6f" % float(prediction[i][0]))
    b = ("%.6f" % float(prediction[i][1]))
    a = round(float(a)*100, 2)
    b = round(float(b)*100, 2)
    c = [a, b]
    predictions.append(c)
print(predictions)

for i in range(len(prediction)):
    photo = test_dataset
    img = photo[0][i]
    if predictions[i][0] < predictions[i][1]:
        answer = 'На изображении Змея, точность:'
        predict = predictions[i][1]
    else:
        answer = 'На изображении Крокодил, точность:'
        predict = predictions[i][0]
    plt.figure()
    plt.imshow(img)
    plt.title(f'{answer} {predict}%')
    plt.show()
