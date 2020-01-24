import os

import torch


def retrain(trainloader, model, device, epoch, criterion, optimizer):
    model.train()

    acc_sum = 0
    loss_sum = 0

    for batch_idx, (image, label) in enumerate(trainloader):
        image, label = image.to(device), label.to(device)

        prediction = model(image)
        loss = criterion(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        # calculate accuracy
        correct = (torch.max(prediction, 1)[1].view(label.size()).data == label.data).sum()
        total = trainloader.batch_size
        train_acc = 100. * correct / total
        acc_sum += train_acc

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tTraining Accuracy: {:.3f}%'.format(
                epoch, batch_idx * len(image), len(trainloader),
                100. * batch_idx / len(trainloader), loss.item(), train_acc))

    acc_avg = acc_sum / len(trainloader)
    loss_avg = loss_sum / len(trainloader)

    result = '\nTrain Epoch: {}\tAverage Loss: {:.3f}\tAverage Accuracy: {:.3f}%'.format(
        epoch, loss_avg, acc_avg)
    print(result)

    os.makedirs('result', exist_ok=True)
    with open('result/train_acc.txt', 'a') as f:
        f.write(str(acc_avg))
    f.close()
    with open('result/train_loss.txt', 'a') as f:
        f.write(str(loss_avg))
    f.close()


def retest(testloader, model, device, criterion, epoch, RESUME):
    model.eval()

    loss_sum = 0
    correct_sum = 0

    for image, label in testloader:
        image, label = image.to(device), label.to(device)

        prediction = model(image)

        # sum up batch loss
        loss = criterion(prediction, label)
        loss_sum += loss.item()

        # get the index of the max log-probability
        pred = prediction.data.max(1, keepdim=True)[1]
        correct = pred.eq(label.data.view_as(pred)).cpu().sum()
        correct_sum += correct

    test_loss = loss_sum / len(testloader.dataset)
    test_acc = 100. * correct_sum / len(testloader.dataset)

    result = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss_sum, correct_sum, len(testloader.dataset), test_acc)
    print(result)

    # Save checkpoint.
    os.makedirs('checkpoint', exist_ok=True)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'checkpoint/' + str(RESUME + int(epoch / 10)) + '.pt')
        with open('result/result.txt', 'a') as f:
            f.write(result)
        f.close()

    os.makedirs('result', exist_ok=True)
    with open('result/test_acc.txt', 'a') as f:
        f.write(str(test_acc))
    f.close()
    with open('result/test_loss.txt', 'a') as f:
        f.write(str(test_loss))
    f.close()
