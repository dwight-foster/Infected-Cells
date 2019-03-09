
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import numpy as np

import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler



train_data = datasets.ImageFolder(r'Drive/Data', transform=transforms.Compose([transforms.RandomResizedCrop(224),transforms.ToTensor()]))
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(0.5 * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16,sampler = train_sampler)
test_loader = torch.utils.data.DataLoader(train_data, batch_size=16,sampler = valid_sampler)

# I wrote a huge CNN but it didn't train as well as the small one. The commented layers mean I didn't use them. 

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,16,3, padding=1)
        self.conv2 = nn.Conv2d(16,4,3,padding=1)
        #self.conv3 = nn.Conv2d(32, 70, 3, padding=1)
        #self.conv4 = nn.Conv2d(70,128,3,padding=1)
        #self.conv5 = nn.Conv2d(128,110,3,padding=1)
        #self.conv6 = nn.Conv2d(110,180, 3, padding=1)
        #self.conv7 = nn.Conv2d(180,200,3,padding=1)
        #self.conv8 = nn.Conv2d(200,230,3,padding=1)
        #self.conv9 = nn.Conv2d(230,290,3,padding=1)
        #self.conv10 = nn.Conv2d(290,330,3,padding=1)
        #self.conv11 = nn.Conv2d(330,400,3,padding=1)
        #self.conv12 = nn.Conv2d(400,600,3,padding=1)
        #self.conv13 = nn.Conv2d(600,450,3,padding=1)
        #self.conv14 = nn.Conv2d(450,300,3,padding=1)
        #self.conv15 = nn.Conv2d(300,60,3,padding=1)
        
        self.pool = nn.MaxPool2d(3,3)
        self.fc1 = nn.Linear(2304,250)
        self.fc2 = nn.Linear(250,200)
        #self.fc3 = nn.Linear(200, 180)
        #self.fc4 = nn.Linear(180,150)
        #self.fc5 = nn.Linear(150,110)
        #self.fc6 = nn.Linear(110,90)
        #self.fc7 = nn.Linear(90,70)
        #self.fc8 = nn.Linear(70,50)
        #self.fc9 = nn.Linear(50,30)
        #self.fc10 = nn.Linear(30,15)
        #self.fc11 = nn.Linear(15,2)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        #x = F.relu(self.pool(self.conv3(x)))
        #x = F.relu(self.pool(self.conv4(x)))
        #x = F.relu(self.pool(self.conv5(x)))
        #x = F.relu(self.pool(self.conv6(x)))
        #x = F.relu(self.pool(self.conv7(x)))
        #x = F.relu(self.pool(self.conv8(x)))
        #x = F.relu(self.pool(self.conv9(x)))
        #x = F.relu(self.pool(self.conv10(x)))
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        #x = self.dropout(F.relu(self.fc2(x)))
        #x = self.dropout(F.relu(self.fc3(x)))
        #x = self.dropout(F.relu(self.fc4(x)))
        #x = self.dropout(F.relu(self.fc5(x)))
        #x = self.dropout(F.relu(self.fc6(x)))
        #x = self.dropout(F.relu(self.fc7(x)))
        #x = self.dropout(F.relu(self.fc8(x)))
        #x = self.dropout(F.relu(self.fc9(x)))
        #x = self.dropout(F.relu(self.fc10(x)))
        x = self.fc2(x)
        return x
model = Net()

model.cuda()      

I used SGD optimizer but if you want to try Adam optimizer you can
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


epochs = 20
for e in range(epochs):
    train_loss = 0.0
    correct = 0.0
    total = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        output = model(images)
        
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        print(train_loss)
        pred = output.data.max(1, keepdim=True)[1]
        correct += np.sum(np.squeeze(pred.eq(labels.data.view_as(pred))).cpu().numpy())
        total += images.size(0)
    print(e)
    print('\nTrain Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
    if correct > max_correct:
        max_correct = correct
        torch.save(vgg16.state_dict(), 'model_transfer.pt')
        print('Saving Model... ')
        
        
 # Here is the test loop
# track test loss

# track test loss
vgg16.load_state_dict(torch.load('model_transfer.pt'))
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))


# iterate over test data
for batch_idx, (images, labels) in enumerate(test_loader):
    # move tensors to GPU if CUDA is available
   
    data, target = images.cuda(), labels.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = vgg16(data)
    
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    pred = output.data.max(1, keepdim=True)[1] 
    # compare predictions to true label
    correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
    total += images.size(0)
    # calculate test accuracy for each object class
print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))  

