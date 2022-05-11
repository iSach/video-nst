import torch.nn.functional as F
import utils as U

def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight, num_style_layers, num_content_layers):
    """
    Computes and returns the style and content losses.
    """
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    
    style_loss = 0
    for name in style_outputs.keys():
        style_loss += F.mse_loss(style_outputs[name], style_targets[name])
    style_loss *= style_weight / num_style_layers

    content_loss = 0
    for name in content_outputs.keys():
        content_loss += F.mse_loss(content_outputs[name], content_targets[name])
    content_loss *= content_weight / num_content_layers

    return style_loss, content_loss


def temporal_loss(prev_img_stylized, target_img, prev_img, curr_img, device):
    """
    Computes the optical-flow-based temporal loss between the previous and current frames.
    """
    # Warp
    warped_img = U.warp_img(prev_img_stylized, prev_img, curr_img, device)

    # Compute temporal loss
    temp_loss = F.mse_loss(warped_img, target_img)
    
    return temp_loss