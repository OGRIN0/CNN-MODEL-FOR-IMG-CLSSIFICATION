import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_iages, testing_labels) = datasets.cifar10.load_data()
training_images, testing_iages = training_images / 255, testing_iages / 255

# The original class_names list had only 8 elements. It should have 10 to match the CIFAR-10 dataset.
class_names = ['plane', 'car', 'bird', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'cat']

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_iages = testing_iages[:40000]
testing_labels = testing_labels[:40000]

model = models.Sequential() # Corrected to models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) # Corrected to Conv2D
model.add(layers.MaxPooling2D((2, 2))) # Corrected to model instead of model1
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Corrected to Conv2D
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Corrected to Conv2D
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # Corrected loss function name

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_iages, testing_labels))

loss,accuracy = model.evaluate(testing_iages, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.keras')
