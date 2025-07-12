import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

COUNT = 5000
EPOCHS = 100 
BATCH = 128
TEST = 0.2

#generating random matrices with points
def randomMatrix(COUNT):

    X = np.zeros((COUNT, 25, 25))
    y = np.zeros(COUNT)
    
    for i in range(COUNT):
        #creating random amount of points
        pointCOUNT = np.random.randint(3, 10)
        points = []
        
        #generating random points and setting them in matrix
        for j in range(pointCOUNT):
            while True:
                point = (np.random.randint(0, 25), np.random.randint(0, 25))
                if X[i, point[0], point[1]] == 0:
                    X[i, point[0], point[1]] = 1
                    points.append(point)
                    break
        
        #calculating all distances, finding minimum and using that as a label for this matrix
        min = 36 #35.something is the max for a 25 by 25 matrix
        for j in range(pointCOUNT):
            for k in range(j+1, pointCOUNT):
                dx = points[j][0] - points[k][0]
                dy = points[j][1] - points[k][1]
                if np.sqrt(dx**2 + dy**2) < min:
                    min = np.sqrt(dx**2 + dy**2)
        #we put the minimum distance we just found here
        y[i] = min
    
    return X, y

#generating the samples to train on
X, y = randomMatrix(COUNT)
#split into 2 sets, train 800 and test 200 sets, test size = 0.2, which is 20% of the whole set, so 200
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=TEST, random_state=42)
#reshape for cnn since it only takes matrices?
XTrain = XTrain.reshape(-1, 25, 25, 1)
XTest = XTest.reshape(-1, 25, 25, 1)

#creating the model and setting its parameters, got from https://www.tensorflow.org/tutorials/images/cnn
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(25, 25, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#trainnig the model, more epochs = less loss, validation about the same throughout, batch size seems to affect processing power, speed, and validation loss for some reason?
history = model.fit(XTrain, yTrain, epochs = EPOCHS, batch_size = BATCH, validation_data=(XTest, yTest))

#checking models accuracy, also got from cnn website
loss, accuracy = model.evaluate(XTest, yTest, verbose=2)
print(f"accuracy: {accuracy}")

#this graph is to show the accuracy of the model
yPrediction = model.predict(XTest).flatten()
plt.figure(figsize=(10, 6))
plt.scatter(yTest, yPrediction, alpha=0.5, label='Predictions')
plt.xlabel("True Min Distance")
plt.ylabel("Predicted Min Distance")
plt.title("True vs Predicted Minimum Distances")
plt.legend()
plt.grid()
plt.savefig(f'B/B_{COUNT}samples_{EPOCHS}epoch_{BATCH}batch_{TEST}test.jpg', bbox_inches='tight')
plt.show()

#this graph is to show the loss of the model, 2 types, validation loss and training loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("epochs")
plt.ylabel("Loss (mean squared error)")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.savefig(f'B/B_{COUNT}samples_{EPOCHS}epoch_{BATCH}batch_{TEST}test LOSSES.jpg', bbox_inches='tight')
plt.show()

#this part is so the user can manually pick an index to look at and see how well the neural network predicted it
while True:
    try:
        #get input from user
        index = int(input(f"0 to exit, Enter a number (0-{COUNT/5} -> test set, {COUNT/5}-{COUNT} -> training set) = "))

        #user chooses to either exit, a matrix from a test set, or one from a training set
        if index == 0:
            break
        if 0 <= index < (COUNT * 0.2):
            matrix = XTest[index].reshape(25, 25)
            true_distance = yTest[index]
            pred_distance = model.predict(XTest[index].reshape(1, 25, 25, 1))[0][0]
            set_type = "Test"
        elif (COUNT * 0.2) <= index < COUNT:
            matrix = XTrain[index-(COUNT * 0.2)].reshape(25, 25)
            true_distance = yTrain[index-(COUNT * 0.2)]
            pred_distance = model.predict(XTrain[index-(COUNT * 0.2)].reshape(1, 25, 25, 1))[0][0]
            set_type = "Train"
        else:
            print("Index out of range. Please enter 0-999.")
            continue
        
        #showing the matrix the user chose
        plt.figure(figsize=(8, 8))
        plt.imshow(matrix, cmap="gray")
        points = np.argwhere(matrix == 1)
        for point in points:
            plt.plot(point[1], point[0])
        plt.title(f"{set_type} Sample {index}\nTrue Distance: {true_distance:.2f}\nPredicted Distance: {pred_distance:.2f}")
        plt.axis('off')
        plt.show()
        
    except ValueError:
        print("enter a valid value")
        break
