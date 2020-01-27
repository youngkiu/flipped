"""
Class Activation Mapping
Googlenet, Kaggle data
"""

import os

import cv2
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from data import *
from inception import inception_v3
from train import *
from update import *

# functions
CAM             = 1
RESUME          = None
PRETRAINED      = False


# hyperparameters
BATCH_SIZE      = 64
IMG_SIZE        = 224
LEARNING_RATE   = 1e-3
EPOCH           = 1000


# prepare data
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
train_data = datasets.ImageFolder(
    '~/dataset/dataset_dogs_vs_cats/train/', transform=transform_train)
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = datasets.ImageFolder(
    '~/dataset/dataset_dogs_vs_cats/test/', transform=transform_test)
testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# fine tuning
if PRETRAINED:
    model = inception_v3(pretrained=PRETRAINED)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(2048, 2)
else:
    model = inception_v3(pretrained=PRETRAINED,
                         num_classes=len(train_data.classes))

model.to(device)


# load checkpoint
if RESUME is not None:
    print("===> Resuming from checkpoint.")
    assert os.path.isfile('checkpoint/'+ str(RESUME) + '.pt'), 'Error: no checkpoint found!'
    model.load_state_dict(torch.load('checkpoint/' + str(RESUME) + '.pt'))


# retrain
criterion = torch.nn.CrossEntropyLoss().to(device)

if PRETRAINED:
    optimizer = torch.optim.SGD(
        model.fc.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
else:
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

writer_train = SummaryWriter('runs/train')
writer_test = SummaryWriter('runs/test')

for epoch in range(EPOCH):
    train_loss, train_acc = retrain(trainloader, model, device, criterion, optimizer, epoch)
    test_loss, test_acc = retest(testloader, model, device, criterion, epoch)

    print('loss - train:%.3f, test:%.3f' % (train_loss, test_loss))
    print('acc  - train:%.3f, test:%.3f' % (train_acc, test_acc))
    writer_train.add_scalar('epoch/loss', train_loss, epoch)
    writer_test.add_scalar('epoch/loss', test_loss, epoch)
    writer_train.add_scalar('epoch/acc', train_acc, epoch)
    writer_test.add_scalar('epoch/acc', test_acc, epoch)


# hook the feature extractor
features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


final_conv = 'Mixed_7c'
model._modules.get(final_conv).register_forward_hook(hook_feature)


# CAM
if CAM:
    root = 'sample.jpg'
    img_pil = Image.open(root)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img_pil)

    # class
    classes = {0: 'cat', 1: 'dog'}

    CAMs = get_cam(model, features_blobs, img_tensor, classes, device)

    img = cv2.imread(root)
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('cam.jpg', result)
