from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.utils import to_categorical
from matplotlib.pyplot import figure, ion, legend, plot, savefig, scatter, show, subplot, subplots, title, xlabel, ylabel
from numpy import argmax, argsort, random, reshape, sqrt, unique
from os import environ
from pandas import read_csv
from seaborn import histplot
# from scipy.linalg import eigh
# from scikeras.wrappers import KerasClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from warnings import filterwarnings
filterwarnings("ignore")
environ["KMP_DUPLICATE_LIB_OK"], environ["TF_CPP_MIN_LOG_LEVEL"], environ["TF_ENABLE_ONEDNN_OPTS"] = "TRUE", "3", "0"

# A class for plotting evaluation of model's performance on test set epoch by epoch
class TestCallback(Callback):
    def __init__(self):
        self.loss, self.acc = [], []

    def on_epoch_end(self, epoch, logs = {}):
        Loss, accuracy = self.model.evaluate(X_test, y_test, verbose = 0)
        self.loss.append(Loss)
        self.acc.append(accuracy)
        # print(f"Testing loss: {Loss}, Accuracy: {accuracy}")

# Loading & normalizing the dataset
trainSet, testSet, Epochs = read_csv("mnist_train.csv"), read_csv("mnist_test.csv"), 30
df, rangeEpochs = trainSet._append(testSet, ignore_index = True), range(Epochs)
X_train, X_test, y_train, y_test, Optimizers, Activations, Initializers, Architecture, learningRates, lossFunction, validationSplit, Accuracy, Loss, validationAccuracy, validationLoss, Title, Alpha, localPopulation, Numbers, numberIterations, numberGenerations, Fitness, Parameters, Tuner = trainSet.drop(["label"], axis = 1).astype("float32") / 255., testSet.drop(["label"], axis = 1).astype("float32") / 255., trainSet["label"].astype("int"), testSet["label"].astype("int"), {Adam: "Adam" , Nadam: "Nadam", SGD: "SGD"}, ["linear", "relu", "tanh"], ["glorot_normal", "glorot_uniform", "zero"], (500, 100, 30), [0.01, 0.1, 0.25, 0.5, 0.85], "binary_crossentropy", 0.2, "accuracy", "loss", "val_accuracy", "val_loss", "Autoencoder Evaluation", 0.6, 1, 10, 1, 1, [], [], {}
numberClasses = len(unique(y_test))
y_train, y_test, dfX_train, localBound, number_test_samples, encoder_input_dimension, decoder_input_dimension, number_hidden_layers, Beta = to_categorical(y_train, numberClasses), to_categorical(y_test, numberClasses), X_train, X_train.shape[1], len(X_test), Architecture[0], Architecture[-1], len(Architecture) - 1, 1 - Alpha
TCB = TestCallback()
# Defining the encoder input & decoder input dimentions.
Image, Length = Input(shape = (localBound,)), int(sqrt(localBound))
arrtestSet = testSet.to_numpy()
print(f"There're {len(X_train)} samples of {Length} X {Length} images as train samples and {number_test_samples} samples as test samples.")

"""
# Plot samples
for Number in range(Numbers):
    fig = figure
    imshow(arrX_train[Number], cmap = "gray")
    savefig(dpi = 1200)
    show()

# Plotting raw data records
Figure, ax = subplots()
ax.pie(df["label"].value_counts(), labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], autopct = '%1.1f%%', shadow = True, startangle = 90)
ax.axis("equal")
savefig(dpi = 1200)
show()
histplot(df["label"])
savefig("MNIST Histogram", dpi = 1200)
show()
"""
"""
def encoderMaker(InputFunction, EncoderArchitecture):
    Encoded = Activation("relu")(Dense(EncoderArchitecture[0])(InputFunction))
    for Index in range(1, len(EncoderArchitecture)):
        Encoded = Activation("relu")(Dense(EncoderArchitecture[Index])(Encoded))
    return Model(InputFunction, Encoded), name = "Encoder")

def decoderMaker(InputFunction, EncoderArchitecture):
    reversedArchitecture = EncoderArchitecture[::-1]
    Decoded = Activation("relu")(Dense(reversedArchitecture[0])(InputFunction))
    for Index in range(1, len(reversedArchitecture)):
        Decoded = Activation("relu")(Dense(reversedArchitecture[Index])(Decoded))
    return Model(InputFunction, Activation("sigmoid")(Dense(localBound)(Decoded)), name = "Decoder")

def autoencoderTrainer(Architecture):
    inputEncoder, inputDecoder = Input(shape = (localBound,), name = "InputEncoder"), Input(shape = (16,), name = "InputDecoder")
    Encoder = encoderMaker(inputEncoder, Architecture)
    Autoencoder = Model(inputEncoder, decoderMaker(inputDecoder, Architecture)(Encoder(inputEncoder)), name = "Autoencoder")    
    Autoencoder.summary()
    Autoencoder.compile(loss = "mse", optimizer = Adam(learning_rate = 0.00025))
    Autoencoder.fit(Train, Train, batch_size = 6000, epochs = 5, shuffle = True)
    return Encoder, Autoencoder

def architectureMutation(Architectures):
    numberArchitectures = len(Architectures)
    for Architecture in Architectures:
        bufferArchitecture = Architecture.copy()
        bufferArchitecture[random.randint(0, len(Architecture))] = random.randint(2, localBound)
        mutatedArchitectures.append(bufferArchitecture)
    return mutatedArchitectures + [list(random.randint(2, localBound, size = random.randint(2, 10))) for Number in range(numberArchitectures)]

# Fine Tuning!
for Iteration in range(numberIterations):
    localArchitecture = list(sort(random.randint(2, localBound, size = random.randint(2, 10))))[::-1]
    _, localAutoencoder = autoencoderTrainer(localArchitecture)
    Architectures.append(localArchitecture)
    Fitness.append(localAutoencoder.evaluate(Validation, Validation))
    Parameters.append(localAutoencoder.count_params())

fsortedIndex = argsort(array(Fitness))[0]
selectedArchitectures, bestArchitecture, bestFitness, bestParameters = [Architectures[val] for val in fsortedIndex[0 : int(localPopulation / 2)]], [Architectures[fsortedIndex]], [Fitness[fsortedIndex]], [Parameters[fsortedIndex]]
for Generation in range(numberGenerations):
    Architectures, Fitness, Parameters = ArchitectureMutation(selectedArchitectures), [], []
    for localArchitecture in Architectures:
        _, localAutoencoder = autoencoderTrainer(localArchitecture)
        fitness.append(localAutoencoder.evaluate(Validation, Validation))
        Parameters.append(localAutoencoder.count_params())

    fsortedIndex = argsort(array(fitness))[0]
    bestArchitecture.append(Architectures[fsortedIndex])
    bestFitness.append(Fitness[fsortedIndex])
    bestParameters.append(Parameters[fsortedIndex])
    selectedArchitectures = [Architectures[val] for val in fsortedIndex[0 : int(localPopulation / 2)] ]

print(bestArchitecture, bestFitness, bestParameters)
"""

# Fine tuning hyperparameters
print("\nTuning autoencoder...")
for Optimizer in Optimizers:
    for learningRate in learningRates:
        for Initializer in Initializers:
            for Activation in Activations:
                # Define & build the encoder & decoder
                decoderInput  = Input(shape = (decoder_input_dimension,))
                Encoded, Decoded = Dense(encoder_input_dimension, activation = Activation, kernel_initializer = Initializer)(Image), Dense(Architecture[-2], activation = Activation, kernel_initializer = Initializer)(decoderInput)
                for hiddenLayer in range(1, number_hidden_layers):
                    Encoded, Decoded = Dense(Architecture[hiddenLayer], activation = Activation, kernel_initializer = Initializer)(Encoded), Dense(Architecture[-(hiddenLayer + 2)], activation = Activation, kernel_initializer = Initializer)(Decoded)

                Encoder, Decoder = Model(Image, Dense(decoder_input_dimension, activation = Activation)(Dense(Architecture[-2], activation = Activation)(Encoded)), name = "Encoder"), Model(decoderInput, Dense(localBound, activation = "sigmoid", kernel_initializer = Initializer)(Decoded), name = "Decoder")
                    
                # Build main autoencoder
                Autoencoder = Model(Image, Decoder(Encoder(Image)))

                # Compile the autoencoder
                Autoencoder.compile(loss = lossFunction, metrics = ["accuracy"], optimizer = Optimizer(learning_rate = learningRate))

                # Fit the data on the autoencoder 500 3
                Autoencoder = Autoencoder.fit(X_train, X_train, batch_size = 60000, epochs = 1, shuffle = True, validation_split = validationSplit, verbose = 0)
                Results = Autoencoder.history
                Tuner[(Optimizer, learningRate, Initializer, Activation)] = Alpha * (Results[validationAccuracy][0] - Results[validationLoss][0]) + Beta * (Results[Accuracy][0] - Results[Loss][0])

Optimizer, learningRate, Initializer, Activation = max(Tuner, key = Tuner.get)
print(f"\n\nOptimal optimizer, primary weights initialization method, activation functions for hidden layers and learning rate for autoencoder neural network are {Optimizers[Optimizer]}, {Initializer}, {Activation} and {learningRate}, respectively.\n\nTraining optimal autoencoder...")

decoderInput = Input(shape = (decoder_input_dimension,))
# Define & build well-tuned encoder & decoder
Encoded, Decoded = Dense(encoder_input_dimension, activation = Activation, kernel_initializer = Initializer)(Image), Dense(Architecture[-2], activation = Activation, kernel_initializer = Initializer)(decoderInput)
for hiddenLayer in range(1, number_hidden_layers):
    Encoded, Decoded = Dense(Architecture[hiddenLayer], activation = Activation, kernel_initializer = Initializer)(Encoded), Dense(Architecture[-(hiddenLayer + 2)], activation = Activation, kernel_initializer = Initializer)(Decoded)

Encoder, Decoder = Model(Image, Dense(Architecture[-1], activation = Activation)(Dense(Architecture[-2], activation = Activation)(Encoded)), name = "Encoder"), Model(decoderInput, Dense(localBound, activation = "sigmoid", kernel_initializer = Initializer)(Decoded), name = "Decoder")

# Build main autoencoder
Autoencoder = Model(Image, Decoder(Encoder(Image)))

# Compile the autoencoder
Autoencoder.compile(loss = lossFunction, metrics = ["accuracy"], optimizer = Optimizer(learning_rate = learningRate))
print(Autoencoder.summary())

# Saving model
Autoencoder.save("Autoencoder.keras")

# Fit the data on the autoencoder
# Test set is now the validation set of model!
autoencoderHistory = Autoencoder.fit(X_train, X_train, batch_size = 500, epochs = 15, shuffle = True, validation_split = validationSplit, verbose = 0)

# Plot the training history (losses)
figure(figsize = (13, 5))
ion()
plot(autoencoderHistory.history[Loss])
plot(autoencoderHistory.history[validationLoss])
title(Title)
ylabel("Loss")
xlabel("Epochs")
legend(["Train Loss", "Validation Loss"], loc = "upper right")
savefig(Title, dpi = 1200)
show()

# Plot the training history (accuracies)
figure(figsize = (13, 5))
ion()
plot(autoencoderHistory.history[Accuracy])
plot(autoencoderHistory.history[validationAccuracy])
title(Title)
ylabel("Accuracy")
xlabel("Epochs")
legend(["Train Accuracy", "Validation Accuracy"], loc = "upper right")
savefig(Title, dpi = 1200)
show()

# Encoding images
print("\nEncoding images...")
X_train, X_test, lossFunction, Title = Encoder.predict(X_train, batch_size = 60000, verbose = 0), Encoder.predict(X_test, batch_size = 1000, verbose = 0), "categorical_crossentropy", "Classifier Evaluation"
print("\n\nTraining the classifier...\n")
# Defining the model
Classifier = Sequential()
Classifier.add(Dense(512, input_dim = 30, kernel_initializer = "uniform", activation = "relu"))
Classifier.add(Dense(256, kernel_initializer = "uniform", activation = "relu"))
Classifier.add(Dense(10, kernel_initializer = "uniform", activation  = "softmax"))

# Compiling the model
Classifier.compile(loss = lossFunction, metrics = ["accuracy"], optimizer = "adam")

print(Classifier.summary())

# Training the model
classifierHistory = Classifier.fit(X_train, y_train, batch_size = 60000, callbacks = [TCB], epochs = Epochs, validation_split = validationSplit, verbose = 0)

# Plotting the training history
figure(figsize = (13, 5))
ion()
plot(classifierHistory.history[Loss])
plot(classifierHistory.history[validationLoss])
plot(rangeEpochs, TCB.loss)
title(Title)
ylabel("Loss")
xlabel("Epochs")
legend(["Train Loss", "Test Loss"], loc = "upper right")
savefig(Title, dpi = 1200)
show()

# Plot the training history (accuracies)
figure(figsize = (13, 5))
ion()
plot(classifierHistory.history[Accuracy])
plot(classifierHistory.history[validationAccuracy])
plot(rangeEpochs, TCB.acc)
title(Title)
ylabel("Accuracy")
xlabel("Epochs")
legend(["Train Accuracy", "Test Accuracy"], loc = "upper right")
savefig(Title, dpi = 1200)
show()

# , Accuracy
Title = "Confusion Matrix"
print(f"\nEvaluating classifier...\n\nModel's score: {Classifier.evaluate(X_test, y_test, batch_size = 1000, verbose = 0)}")
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, argmax(Classifier.predict(X_test, batch_size = 1000, verbose = 0), axis = 1))).plot()
title(Title)
savefig(Title, dpi = 1200)
show()
"""
model = KerasClassifier(model = model, batch_size = 60000, epochs = 1, verbose = 0)
model.fit(X_train, y_train, validation_data = (X_test, y_test))
y_pred = model.predict(X_test)
"""
