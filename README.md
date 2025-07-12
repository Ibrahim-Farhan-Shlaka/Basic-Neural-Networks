# Basic-Neural-Networks
A collection of 5 basic neural networks that are trained to do different tasks on binary matrices, the tasks being findind the distance between two points, finding the closest two point distance, finding the furthest two point distance, finding the amount of points, and finding the amount of squares in a binary matrix.
All were made using tensorflow in python.

# Parameters
1. **Count** : amount of sample matrices made to train the model on.
2. **Epoch** : amount of times the model is trained on the samples.
3. **Batch** : amount of samples processed by the model at once.
4. **Test** :  percentage of the samples used for testing, the rest are left for training the model (0.2 = 20%, 0.5 = 50% etc.).

# Types
1. **Distance** : The distance model is trained on matrices with only two points in them at all times, the label of each sample is the euclidian distance between points, the model guesses the distance between 2 points in a given test matrix.
2. **Closest Distance** : The closest distance model is trained on matrices with 2-10 points in them, the label of each sample is the minimum euclidian distance between the closest 2 points, the model guesses the distance between the 2 closest points in a given test matrix.
3. **Furthest Distance** : The furthest distance model is trained on matrices with 2-10 points in them, the label of each sample is the maximum euclidian distance between the closest 2 points, the model guesses the distance between the 2 furthest points in a given test matrix.
4. **Point Count** : The point count model is trained on matrices with 2-10 points in them, the label of each sample is the amount of points in each matrix, the model guesses the amount of points in a given test matrix.
5. **Square Count** : The square count model is trained on matrices with 2-10 squares that may or may not intersect with eachother, the label of each sample is the amount of squares in each matrix, the model guesses the amount of squares in a given matrix.

# Notes
1. the models all output graphs for the sake of statistical analysis, to help you tune the parameters to your liking.

# Thank you :)
