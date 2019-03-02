# Infected-Cells
Classifying infected vs. uninfected malaria cells.

CNN from scratch is just a 3 convolutional layer and 2 fully connected layer cnn. It is small but when I tried a larger one it didn't work as well. 

I used a vgg16 model for transfer learning. I changed the output nodes to 2 and got 78% accuracy for testing and training. 

## Training

I started with a learning rate of 0.01 for 20 epochs. Then when my model had trained I went down to about 0.003 and so on 

for about 4 more cycles of that and got around 70% acurracy on the training data. 

## Installing 
I used this [kaggle](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria) dataset.
Requirements are Pytorch and Numpy

## GPU
I used google Colab GPU because I don't have a good computer. It trained 20 epochs in around 10 minutes. 
