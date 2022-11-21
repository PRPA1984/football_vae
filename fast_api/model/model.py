import torch
import torchvision
import numpy as np
from torchvision import transforms

preprocess = transforms.Resize(64)


def remove_noise(image):
    alpha = image[3, :,:]> 50
    alpha = alpha.type(torch.uint8)
    noise_filtered = torch.mul(alpha, image)
    return noise_filtered[:3,:,:]

def get_image_by_id(id):
    im = remove_noise(torchvision.io.read_image(f"Images/{id}.png"))
    return preprocess(im.type(torch.float32)) / 255

def get_interp(v1, v2, n):
  if not v1.shape == v2.shape:
    raise Exception('Different vector size')

  return np.array([np.linspace(v1[i], v2[i], n+2) for i in range(v1.shape[0])]).T



def model_interp(model, index1, index2, size = 10):

    image1 = get_image_by_id(index1).unsqueeze(0)
    image2 = get_image_by_id(index2).unsqueeze(0)

    img1_compressed,_,_ = model.encoder(image1.to("cpu"))
    img2_compressed,_,_ = model.encoder(image2.to("cpu"))

    interp = get_interp(img1_compressed.detach().cpu(), img2_compressed.detach().cpu(), size)

    interp = torch.from_numpy(interp)

    interp = interp.permute(1,0,2).unsqueeze(3)

    print(interp.shape)

    artificial_images = model.decoder(interp.to("cpu"))

    return artificial_images


def random_player(model):
  rand_vec = torch.randn((1,500,1,1))
  return model.decoder(rand_vec.to("cpu")).squeeze(0)




