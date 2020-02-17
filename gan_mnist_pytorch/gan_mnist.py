# https://github.com/YBIGTA/Deep_learning/blob/master/GAN/2017-07-29-GAN-tutorial-2-MNIST.markdown

import itertools
import math
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from IPython import display
from torch.autograd import Variable

transform = transforms.Compose([
    transforms.ToTensor(),
    # https://stackoverflow.com/questions/55124407/output-and-broadcast-shape-mismatch-in-mnist-torchvision
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

train_dataset = dsets.MNIST(root='./data/', train=True,
                            download=True, transform=transform)  # 전체 training dataset
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=100, shuffle=True)  # dataset 중에서 샘플링하는 data loader입니다


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # MNIST 이미지 사이즈: 28*28 => 일렬로 늘이면 784인 벡터 => 얘를 input으로 받아 1024로 확장
            nn.Linear(784, 1024),
            # Leaky ReLU => x if x>0 else a*x, 여기서 a는 0.2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),                      # 30% drop out

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()                          # 시그모이드를 통해 "진짜 이미지일 확률" 추출
        )

    def forward(self, x):
        out = self.model(x.view(x.size(0), 784))  #
        out = out.view(out.size(0), -1)           # 배치사이즈 * 1의 벡터가 결과값으로 나오게 됨
        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # size가 100인 noise를 인풋으로 받는다, 가중치 벡터를 곱해 256차원 벡터로 확장
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2, inplace=True),      # Leaky ReLU

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), 100)
        out = self.model(x)
        return out


discriminator = Discriminator().cuda()
generator = Generator().cuda()


criterion = nn.BCELoss()
lr = 0.0002
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)


def train_discriminator(discriminator, images, real_labels, fake_images, fake_labels):
    discriminator.zero_grad()                     # parameter를 0으로 초기화
    outputs = discriminator(images)               # 실제 image들을 분류기에 넣고 돌린 결과(진짜인지 아닌지) -  1:진짜, 0:가짜
    real_loss = criterion(outputs, real_labels)   # output과 실제 label(1이겠죠)과 비교하여 loss 계산()
    real_score = outputs                          # output의 리스트를 real_score라는 변수에 저장..

    outputs = discriminator(fake_images)          # fake 이미지들을 분류기에 넣고 돌린 결과
    fake_loss = criterion(outputs, fake_labels)   # output과 label(0이겠죠)과 비교하여 loss 계산()
    fake_score = outputs                          # output의 리스트를 fake_score라는 변수에 저장..

    d_loss = real_loss + fake_loss                # descriminator의 loss는 두 loss를 더한 것
    d_loss.backward()                             # 오차 역전파
    d_optimizer.step()                            # parameter update
    return d_loss, real_score, fake_score


def train_generator(generator, discriminator_outputs, real_labels):
    generator.zero_grad()                                     # parameter 0으로 초기화
    g_loss = criterion(discriminator_outputs, real_labels)    # loss 계산 -> Discriminator에 fake-data를 넣었을 때 결과와
                                                              # 실제 레이블(1)을 비교, 이 부분이 핵심
    g_loss.backward()                                         # 역전파
    g_optimizer.step()                                        # parameter update
    return g_loss


num_test_samples = 16
test_noise = Variable(torch.randn(num_test_samples, 100).cuda())


# 이 부분은 이미지를 jupyter notebook에서 보여주기 위한 초기 세팅입니다

size_figure_grid = int(math.sqrt(num_test_samples))
fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6, 6))
for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
    ax[i, j].get_xaxis().set_visible(False)
    ax[i, j].get_yaxis().set_visible(False)

# 에폭과 배치 개수를 설정합니다.

num_epochs = 200
num_batches = len(train_loader)
num_fig = 0

# Discriminator와 Generator의 Loss, 그리고 Discriminator의 real image에 대한 스코어, fake image에 대한 스코어를 tracking합시다.

tracking_dict = {}
tracking_dict["d_loss"] = []
tracking_dict["g_loss"] = []
tracking_dict["real_score"] = []
tracking_dict["fake_score"] = []

# 자 이제 진짜 시작해봅시다.

for epoch in range(num_epochs):
    for n, (images, _) in enumerate(train_loader):
        # real images를 pytorch 변수로 바꿔줍니다
        images = Variable(images.cuda())
        # real images의 labels를 pytorch 변수로 바꿔줍니다.
        real_labels = Variable(torch.ones(images.size(0)).cuda())

        # 샘플링

        # Generator의 인풋값인 noise를 추출한다.
        noise = Variable(torch.randn(images.size(0), 100).cuda())
        # generator에 noise를 넣어 fake image를 만든다
        fake_images = generator(noise)
        # fake images의 labels를 가져옵니다.
        fake_labels = Variable(torch.zeros(images.size(0)).cuda())

        # Discriminator를 학습시킵니다

        d_loss, real_score, fake_score = train_discriminator(
            discriminator, images, real_labels, fake_images, fake_labels)

        # Discriminator를 새로운 fake images에 대해 테스트해봅니다.

        noise = Variable(torch.randn(images.size(0), 100).cuda())
        fake_images = generator(noise)
        outputs = discriminator(fake_images)

        # 테스트 결과와 실제 레이블(영벡터..)을 비교하여 generator를 학습시킵니다.

        g_loss = train_generator(generator, outputs, real_labels)

        # 100번째 마다 test_image를 확인해볼껍니다.

        if (n+1) % 100 == 0:
            test_images = generator(test_noise)

            # 이미지를 쥬피터 노트북에 띄웁니다.

            for k in range(num_test_samples):
                i = k//4
                j = k % 4
                ax[i, j].cla()
                ax[i, j].imshow(test_images[k, :].data.cpu(
                ).numpy().reshape(28, 28), cmap='Greys')
            display.clear_output(wait=True)
            display.display(plt.gcf())

            plt.savefig('results/mnist-gan-%03d.png' % num_fig)
            num_fig += 1
            # https://github.com/NVIDIA/flownet2-pytorch/issues/113#issuecomment-450802359
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, '
                  'D(x): %.2f, D(G(z)): %.2f'
                  % (epoch + 1, num_epochs, n+1, num_batches, d_loss.data, g_loss.data,
                     real_score.data.mean(), fake_score.data.mean()))
            tracking_dict["d_loss"].append(d_loss.data)
            tracking_dict["g_loss"].append(g_loss.data)
            tracking_dict["real_score"].append(real_score.data.mean())
            tracking_dict["fake_score"].append(fake_score.data.mean())
