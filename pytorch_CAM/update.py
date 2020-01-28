import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms


# generate class activation mapping for the top1 prediction
def returnCAM(feature_conv, weight_softmax, class_idx):
    # feature_conv.shape = (1, 2048, 5, 5)
    # weight_softmax.shape = (2, 2048)

    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for i, idx in enumerate(class_idx):
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        # cam.shape = (1, 25)
        cam = cam.reshape(h, w)
        # cam.shape = (5, 5)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        cam_img_upsample = cv2.resize(cam_img, size_upsample)
        # cam_img_upsample.shape = (256, 256)
        output_cam.append(cam_img_upsample)

        if i < 2:
            plt.subplot(1, 2, i+1)
            plt.title(idx)
            plt.imshow(cam_img_upsample)

    plt.show(block=False)
    plt.pause(1)
    plt.savefig('sorted_class_cam.png')

    return output_cam


def get_cam(model, features_blobs, img_tensor, classes, device):
    # img_tensor.shape = torch.Size([3, 224, 224])
    img_variable = img_tensor.unsqueeze(0).to(device)
    # img_variable.shape = torch.Size([1, 3, 224, 224])
    logit = model(img_variable)
    # logit.data = tensor([[ 0.2154, -1.5496]]), logit.shape = torch.Size([1, 2])

    h_x = F.softmax(logit, dim=1).data.squeeze()
    # h_x.data = tensor([0.8538, 0.1462]), h_x.shape = torch.Size([2])
    probs, idx = h_x.sort(0, True)
    # probs.data = tensor([0.8538, 0.1462])
    # idx.data = tensor([0, 1])

    # output: the prediction
    for i in range(len(classes)):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i].item()]))

    params = list(model.parameters())
    # len(params) = 292
    # params[-6] == params[286], params[286].shape = torch.Size([384])
    # params[-5] == params[287], params[287].shape = torch.Size([192, 2048, 1, 1])
    # params[-4] == params[288], params[288].shape = torch.Size([192])
    # params[-3] == params[289], params[289].shape = torch.Size([192])
    # params[-2] == params[290], params[290].shape = torch.Size([2, 2048])
    # params[-1] == params[291], params[291].shape = torch.Size([2])
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
    # weight_softmax.shape = (2, 2048)

    CAMs = returnCAM(features_blobs[0],
                     weight_softmax, idx.data.cpu().tolist())

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' %
          classes[idx[0].item()])

    return CAMs
