import numpy as np
import cv2
import os
from glob import glob

import torch
import torch.nn.functional as F
import torch.optim as optim

from models import models

device = torch.device("cuda")

model = models["256"].to(device)

learning_rate = 0.0001
learning_rate_decay = 0.95
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = F.mse_loss

log_interval = 1
save_interval = 5

module_name = "cell_fluoresce_10x"


def train(train_data):
    model.train()
    for batch_idx, (input, target) in enumerate(train_data):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print("\tRate:{} \tLoss: {:.6f}".format(learning_rate, loss.item()))


if __name__ == "__main__":
    train_dir = os.path.join("modules", module_name, "dataset")
    train_data = []
    batch_size = 50
    epochs = 250

    input_list = sorted(glob(train_dir + "/input/*"))

    for input_filepath in input_list:
        input_file = os.path.basename(input_filepath)
        output_file = input_file.replace("input", "output")
        output_filepath = os.path.join(train_dir, "output", output_file)

        if os.path.isfile(output_filepath):
            input_img = cv2.resize(cv2.imread(input_filepath), (512, 512))
            output_img = cv2.resize(cv2.imread(output_filepath), (512, 512))

            # cv2.imshow("input", input_img)
            # cv2.imshow("output", output_img)
            # cv2.waitKey()

            input_img = input_img[:, :, ::-1]
            output_img = output_img[:, :, ::-1]

            input_img = np.swapaxes(np.swapaxes(np.array(input_img, dtype=float), 0, 2), 1, 2) / 255.0
            output_img = np.swapaxes(np.swapaxes(np.array(output_img, dtype=float), 0, 2), 1, 2) / 255.0

            input_shape = (1,) + input_img.shape
            output_shape = (1,) + output_img.shape

            input_cuda = torch.from_numpy(input_img.reshape(input_shape)).float().cuda()
            output_cuda = torch.from_numpy(output_img.reshape(output_shape)).float().cuda()

            train_data.append((input_cuda, output_cuda))
    train_data = train_data[0:10]

    for epoch in range(1, epochs + 1):
        print("Epoch: %d" % epoch)

        batch_cnt = len(train_data) / batch_size

        for batch_idx in range(0, len(train_data), batch_size):
            print("[Images %d-%d of %d]" % (batch_idx, batch_idx + batch_size, batch_cnt))
            train(train_data[batch_idx:batch_idx + batch_size])

        # print(model.state_dict())
        if epoch > 0 and epoch % save_interval == 0:
            torch.save(model.state_dict(), os.path.join("modules", module_name, "checkpoints", "save_{:03d}.pth".format(epoch)))

    torch.save(model.state_dict(), os.path.join("modules", module_name, "weights.pth"))
