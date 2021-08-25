from torch import nn

# the input size is (150 * 150 * 3)
input_size = (100, 100, 3)


# class of the model to train and use
# the CNN type is used to design a model in this case
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv2d -> conv2d (-> conv2d if it is needed) -> flatten -> fc layer (-> fc layer if it is needed)
        # conv2d - relu function & fc layer - softmax function
        pass  # TODO implement this

    def forward(self, x):
        pass  # TODO implement this
