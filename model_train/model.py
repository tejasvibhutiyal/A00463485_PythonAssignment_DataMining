from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input
from tensorflow.keras.utils import to_categorical

def create_model():
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

# Function to load and preprocess the MNIST data
def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # Reshape to (28, 28, 1), normalize and one-hot encode labels
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return (train_images, train_labels), (test_images, test_labels)

# Function to create the model architecture


# Function to train the model
def train_model():
    (train_images, train_labels), (test_images, test_labels) = load_data()
    model = create_model()
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Change this line
              metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=50, batch_size=128, validation_split=0.1)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc}")
    return model

# Function to save the model to disk
def save_model(model, filename='digit_classifier_model.h5'):
    model.save(filename)
    print(f"Model saved to {filename}")

# Main function to handle model creation and training
if __name__ == '__main__':
    model = train_model()
    save_model(model)
