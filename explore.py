# %%
import torch
import numpy as np
import matplotlib.pyplot as plt


x = torch.rand(1, 3, 16, 16)
print('input tensor x=')
print(x)
# %%
# visualize x
x_array = np.array(x)[0,:,:,:]  # shape = [3,16,16]

plt.axis("off")
plt.imshow(x_array.transpose((1,2,0)))  # shape = [16,16,3]
plt.show()
# %%
# CNN model
import torch.nn.functional as F


class CNN(torch.nn.Module):

    def __init__(self) -> None:
        super(CNN, self).__init__()

        self.conv = torch.nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1)

    def forward(self, x):
        x= F.relu(self.conv(x))
        return x

cnn = CNN()

print('\n\nModel params:')
for param in cnn.parameters():
    print(param)

# %%
# correct input index: [B,C,W,H]
y = cnn(x)  # shape = [1,3,16,16]
print('\n\n', y.shape)

# %%
# incorrect input index: [B,W,H,C]
try:
    x_BWHC = x.permute(0,2,3,1)  # shape = [1,16,16,3]
    y_BWHC = cnn(x_BWHC)
except RuntimeError as e:
    print("\n\nRuntimeError:", e)  # error

# %%
