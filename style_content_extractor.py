import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models.vgg import vgg19

"""
This module is used to extract the style and content features from an image.

It is based on a pre-trained version of VGG19, included in PyTorch (TorchVision).

Note that the network is frozen during the whole proccess!
"""
class StyleContentExtractor(nn.Module):
    def __init__(self, style_layers, content_layers, device):
        super(StyleContentExtractor, self).__init__()
        self.vgg = vgg19(pretrained=True).features.to(device).eval()
        self.layers = style_layers + content_layers
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_style_layers = len(style_layers)
        self.num_layers = len(style_layers) + len(content_layers)
        self.pre_transform = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def gram_matrix(self, input):
        _, b, c, d = input.size()
        features = input.view(b, c * d)
        G = torch.mm(features, features.t())
        return G.div(b * c * d).unsqueeze(0)

    def extract_features(self, x):
        block_index = 1
        conv_index = 1
        layers_outputs = []
        extracted = 0
        for f in self.vgg.children():
            if extracted == self.num_layers:
                break
            
            x = f(x)
            
            f_name = f.__class__.__name__
            if 'Conv2d' in f_name:
                f_type = 'conv'
            elif 'MaxPool2d' in f_name:
                f_type = 'pool'
            elif 'ReLU' in f_name:
                f_type = 'relu'

            name = 'block' + str(block_index) + '_' + f_type + str(conv_index)
            if name in self.layers:
                layers_outputs.append(x.clone())
                extracted += 1

            if f_type == 'pool':
                block_index = block_index + 1
                conv_index = 1
            elif f_type == 'conv':
                conv_index = conv_index + 1

        return layers_outputs

    def forward(self, x):
        x = self.pre_transform(x)
        outputs = self.extract_features(x)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_outputs = [self.gram_matrix(style_output)
                            for style_output in style_outputs]

        style_dict = {style_name: value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}