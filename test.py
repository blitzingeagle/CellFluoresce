import numpy as np
import cv2
import os
import torch

from models import models

device = torch.device("cuda")

model = models["256"].to(device)

module_path = "modules/cell_fluoresce_10x"
weights = torch.load(os.path.join(module_path, "checkpoints-256x256-256-mse_loss/save_250.pth"))
model.load_state_dict(weights)


def img_to_input(img):
    img = img[:, :, ::-1]
    img = np.swapaxes(np.swapaxes(np.array(img / 255., dtype=np.float32), 0, 2), 1, 2)
    shape = (1,) + img.shape
    return torch.from_numpy(img.reshape(shape)).float().cuda()


def output_to_img(output):
    img = output.cpu().detach().numpy()
    img = img.reshape(img.shape[1:])
    img = np.swapaxes(np.swapaxes(np.array(img * 255., dtype=np.uint8), 0, 2), 0, 1)
    return img[:, :, ::-1]


if __name__ == "__main__":
    model.eval()

    with torch.no_grad():
        for x in range(1, 21):
            input_img = cv2.resize(cv2.imread(os.path.join(module_path, "dataset/input/%d.jpg" % x)), (256, 256))
            input_cuda = img_to_input(input_img)

            output_cuda = model(input_cuda)
            output_img = output_to_img(output_cuda)

            target_img = cv2.resize(cv2.imread(os.path.join(module_path, "dataset/output/%d.jpg" % x)), (256, 256))
            target_img = np.array(target_img, dtype=np.uint8)

            cv2.imshow("input", input_img)
            cv2.moveWindow("input", 0, 0)
            cv2.imshow("output", output_img)
            cv2.moveWindow("output", 400, 0)
            cv2.imshow("target", target_img)
            cv2.moveWindow("target", 800, 0)

            cv2.waitKey()

            # cv2.destroyAllWindows()
