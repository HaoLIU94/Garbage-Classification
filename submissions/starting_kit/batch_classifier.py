
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms.functional as TF
import time
import torchvision
from PIL import Image


"""
Transformation of data is done in this file because data generator needs to keep the same
type of data as in the beginning, and those transformation requires the data being an image
which is incompatible.
Other transformations on numpy array can be done in the image_preprocessor file.
"""
def my_transforms(image):
    """
    Those are some simple transformation to have a correct size of inputs.
    You can add any transformation you want but keep in mind that the return should be only
    one input.
    """
    scale = 360
    input_shape = 224

    # images = []
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = TF.resize(image, scale)
    image = TF.resized_crop(image, i=68, j=68, h=input_shape, w=input_shape, size=input_shape)
    return image
    """
    Exemples of possible transformations :
    TF.hflip(image))
    TF.vflip(image))
    TF.rotate(image, 90))
    TF.rotate(image, 270))
    """

def data_transforms(inputs):
    """
    You can change mean & std but the rest shouldn't be except if you to change the 
    size of input in your NN
    """
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    res = torch.empty(inputs.shape[0], 3, 224, 224)
    for i, img in enumerate(inputs):
        x = my_transforms(img)
        x = TF.to_tensor(x)
        x = TF.normalize(x, std, mean)
        res[i] = x
    return res


class BatchClassifier(object):
    """
    gen_train, gen_valid... shouldn't be modified
    You can change the rest like criterion, optimizer or even the function _build_model if you want to
    
    Accuracy on train and valid during the epochs are on data that are part of the train data for the
    RAMP challenge, so you shouldn't expect to see the same values.
    """
    def __init__(self):
        self.model = self._build_model()
        self.epochs = 1

    def fit(self, gen_builder):
        batch_size = 64

        gen_train, gen_valid, nb_train, nb_valid =\
            gen_builder.get_train_valid_generators(
                batch_size=batch_size, valid_ratio=0.3)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=0.001, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        since = time.time()
        use_gpu = False

        # graphs
        loss_values_train = []
        loss_values_valid = []
        acc_values_train = []
        acc_values_valid = []

        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch, self.epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    self.model.train(True)  # Set model to training mode
                else:
                    self.model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                gen_data = None
                nums = 0
                if phase == 'train':
                    gen_data = gen_train
                    nums = nb_train
                else:
                    gen_data = gen_valid
                    nums = nb_valid

                num_exemples = 0
                while nums > 0:
                    nums -= batch_size
                    data = next(gen_data)

                    inputs, labels = data

                    inputs = data_transforms(inputs)
                    labels = torch.from_numpy(labels)
                    labels = labels.type(torch.LongTensor)

                    num_exemples += labels.size()[0]

                    if use_gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = self.model(inputs)
                    if type(outputs) == tuple:
                        outputs, _ = outputs
                    _, preds = torch.max(outputs, 1)
                    _, labels = torch.max(labels, 1)
                    loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item()
                    running_corrects += int(torch.sum(preds == labels))

                epoch_loss = running_loss / num_exemples
                epoch_acc = running_corrects / num_exemples

                if phase == 'train':
                    loss_values_train.append(epoch_loss)
                    acc_values_train.append(epoch_acc)
                else:
                    loss_values_valid.append(epoch_loss)
                    acc_values_valid.append(epoch_acc)


                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))


    def predict_proba(self, X):
        X = data_transforms(X)
        outputs = self.model(X)
        res = []
        if type(outputs) == tuple:
            outputs, _ = outputs
        _, preds = torch.max(outputs, 1)
        for p in preds:
            p_prob = [0,0,0,0,0,0]
            p_prob[p] = 1
            res.append(p_prob)
        return res

    def _build_model(self):
        model_conv = torchvision.models.resnet18(pretrained=True)
        n_class = 6 # Number of filters in the bottleneck layer
        # Since imagenet has 1000 classes , We need to change our last layer according to 
        # the number of classes we have
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, n_class)
        return model_conv
