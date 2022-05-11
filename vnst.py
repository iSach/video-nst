from style_content_extractor import StyleContentExtractor
import sys
import getopt
import utils as U
import torch
import torch.optim as optim
from torchvision import utils, transforms as T
from tqdm import tqdm
from losses import *
from PIL import Image
import matplotlib.pyplot as plt

content_layers = ['block4_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

def apply_style_transfer(content_path, style_path, output_path, content_weight,
                         style_weight, resize, temporal_loss, 
                         temporal_weight, target_start, start_from_prev, 
                         style_from_prev, opt, iters, early_stopping):

    # Load device
    cuda_on = torch.cuda.is_available()
    device = 'cuda' if cuda_on else 'cpu'
    print('Device:', 
          device, 
          '(' + torch.cuda.get_device_name(0) +')' if cuda_on else '')

    if cuda_on:
        torch.cuda.empty_cache()

    # Determine image or video.
    nst_mode = None  # video or image
    extension = content_path.split('.')[1]
    if extension == 'mp4' or extension == 'gif':
        nst_mode = 'video'
    elif extension == 'jpg' or extension == 'png':
        nst_mode = 'image'
    else:
        print('Unknown file extension: {}'.format(extension))
        sys.exit(2)

    # Announce
    print('-' * 80)
    print('Applying style transfer...')
    print('Content {}: {}'.format(nst_mode, content_path.split('/')[-1]))
    print('Style image: {}'.format(style_path.split('/')[-1]))
    print('Optimizer:', opt)
    print('Content weight: {}'.format(content_weight))
    print('Style weight: {}'.format(style_weight))
    print('Target start: {}'.format(target_start))
    print('Early stopping: {}'.format(early_stopping))
    if nst_mode == 'video':
        print('Temporal loss: {}'.format(temporal_loss))
        print('Temporal weight: {}'.format(temporal_weight))
        print('Optimize from warped previous frame: {}'.format(start_from_prev))
        print('Style from previous frame: {}'.format(style_from_prev))

    # Start style transfer
    if nst_mode == 'image':
        image_style_transfer(content_path, style_path, output_path, 
                                      content_weight, style_weight, resize, 
                                      target_start, iters, early_stopping, opt, device)
    else:
        video_style_transfer(content_path, style_path, output_path, 
                                      content_weight, style_weight, resize, 
                                      temporal_loss, temporal_weight, target_start, 
                                      start_from_prev, style_from_prev, 
                                      iters, early_stopping, opt, device)


def image_style_transfer(content_path, style_path, output_path, content_weight,
                         style_weight, resize, target_start,
                         iters, early_stopping, opt, device):

    # Initialize StyleContentExtractor
    sce = StyleContentExtractor(style_layers, content_layers, device)

    # Load images
    content_img = U.load_image(content_path, resize, device)
    style_img = U.load_image(style_path, resize, device)

    output_img = __transfer(sce, content_img, style_img, content_weight,
                            style_weight, False,
                            0, target_start, None, None, False, iters, 
                            early_stopping, opt, device)
    
    print('Image generation done. Saving to {}...'.format(output_path))
    utils.save_image(output_img, output_path)


def video_style_transfer(content_path, style_path, output_path, 
                         content_weight, style_weight, resize, 
                         temporal_loss, temporal_weight, target_start, 
                         start_from_prev, style_from_prev,
                         iters, early_stopping, opt, device):

    # Initialize StyleContentExtractor
    sce = StyleContentExtractor(style_layers, content_layers, device)

    content_gif = Image.open(content_path)
    style_img = U.load_image(style_path, resize, device)

    transferred_frames = []
    prev_frame = None
    prev_frame_stylized = None
    opti_bar = tqdm(total=iters, position=0, leave=True)
    pbar = tqdm(total=int(content_gif.n_frames), position=1, leave=True)
    for frame in range(0, int(content_gif.n_frames)):
        content_gif.seek(frame)
        content_img = U.process_image(content_gif.convert('RGB'), resize, device)
        transferred_img = __transfer(sce, content_img, style_img, content_weight,
                                     style_weight, temporal_loss, 
                                     temporal_weight, target_start, 
                                     prev_frame_stylized, prev_frame,
                                     start_from_prev, iters, early_stopping, opt, 
                                     device, p_bar=opti_bar)
        prev_frame = content_img
        prev_frame_stylized = transferred_img
        if style_from_prev:
            style_img = transferred_img.squeeze(0).clone().detach()
        transferred_frames.append(T.ToPILImage()(transferred_img.squeeze(0)))
        pbar.update(1)

    opti_bar.close()
    pbar.close()
    
    print('Video generation done. Saving to {}...'.format(output_path))
    transferred_frames[0].save(output_path, save_all=True, 
                               append_images=transferred_frames[1:], loop=0)


def __transfer(sce, content_img, style_img, content_weight,
               style_weight, temp_loss, temporal_weight, 
               target_start, prev_img_stylized, prev_frame,
               start_from_prev, iters, early_stopping, opt, device, p_bar=None):
    style_targets = sce(style_img.detach().unsqueeze(0))['style']
    content_targets = sce(content_img.detach().unsqueeze(0))['content']

    if target_start == 'content':
        target_img = content_img.unsqueeze(0).clone()
    elif target_start == 'random':
        target_img = torch.randn(content_img.unsqueeze(0).data.size(), 
                                 device=device)
    
    if prev_frame is not None:
        prev_frame = prev_frame.detach()
    
    if prev_img_stylized is not None:
        prev_img_stylized = prev_img_stylized.detach()

    if start_from_prev and prev_img_stylized is not None:
        target_img = U.warp_img(prev_img_stylized, prev_frame, content_img, device).contiguous()

    target_img.requires_grad_(True)
    sce.requires_grad_(False)
    sce.vgg.requires_grad_(False)

    pbar = tqdm(total=iters) if p_bar is None else p_bar
    pbar.reset()
    pbar.refresh()

    s_losses = []
    c_losses = []
    temp_losses = []
    losses = []
    
    if opt == 'adam':
        optimizer = optim.Adam([target_img], lr=0.02, betas=(0.99, 0.999), eps=1e-1)
        while pbar.n < iters:
            with torch.no_grad():
                target_img.clamp_(0, 1)

            optimizer.zero_grad()
            outputs = sce(target_img)

            # Compute total loss
            style_loss, content_loss = style_content_loss(outputs, style_targets, 
                                    content_targets, style_weight, 
                                    content_weight, 5, 1)
            loss = style_loss + content_loss

            with torch.no_grad():
                s_losses.append(style_loss.detach().cpu().numpy())
                c_losses.append(content_loss.detach().cpu().numpy())

            if temp_loss and prev_img_stylized is not None:
                temploss = temporal_weight * temporal_loss(prev_img_stylized, target_img, 
                                                        prev_frame, content_img, device)
                loss += temploss
                temp_losses.append(temploss.detach().cpu().numpy())

            with torch.no_grad():
                losses.append(loss.detach().cpu().numpy())

                if pbar.n > 50 and early_stopping:
                    last_50_loss = losses[-50]
                    if last_50_loss > 0 and torch.abs(last_50_loss - loss) / loss < 1e-3:
                        break
            
            loss.backward(retain_graph=True)
            pbar.update(1)

            optimizer.step()
    elif opt == 'lbfgs':
        optimizer = optim.LBFGS([target_img])
        while pbar.n < iters:
            def closure():
                with torch.no_grad():
                    target_img.clamp_(0, 1)

                optimizer.zero_grad()
                outputs = sce(target_img)

                # Compute total loss
                style_loss, content_loss = style_content_loss(outputs, style_targets, 
                                        content_targets, style_weight, 
                                        content_weight, 5, 1)
                loss = style_loss + content_loss

                with torch.no_grad():
                    s_losses.append(style_loss.detach().cpu().numpy())
                    c_losses.append(content_loss.detach().cpu().numpy())

                if temp_loss and prev_img_stylized is not None:
                    temploss = temporal_weight * temporal_loss(prev_img_stylized, target_img, 
                                                            prev_frame, content_img, device)
                    loss += temploss
                    temp_losses.append(temploss.detach().cpu().numpy())

                with torch.no_grad():
                    losses.append(loss.detach().cpu().numpy())

                    # early stopping not available for LBFGS.

                loss.backward(retain_graph=True)
                pbar.update(1)
                
                return loss
            optimizer.step(closure)

    with torch.no_grad():
        target_img.clamp_(0, 1)

    # toggle loss plots
    if False:
        plt.plot(losses, label='Loss')
        plt.legend()
        plt.show()
        plt.plot(s_losses, label='Style Loss')
        plt.legend()
        plt.show()
        plt.plot(c_losses, label='Content Loss')
        plt.legend()
        plt.show()
        if temp_loss:
            plt.plot(temp_losses, label='Temporal Loss')
            plt.legend()
            plt.show()

    return target_img

        
def main(argv):
    arg_help = "{0} -c content -s style [-o output ".format(argv[0]) + \
        "-w content_weight -y style_weight -z size -T true/false " + \
        "-W temporal_weight -i iters" + \
        "-I content/random -p true/false -Y true/false -a lbfgs/adam -e true/false] "

    try:
        opts, args = getopt.getopt(argv[1:], 
                                   "hc:s:o:w:y:z:T:W:i:I:p:Y:a:e:",
                                   ["help", 
                                    "content=", 
                                    "style=", 
                                    "output=",
                                    "content_weight=",
                                    "style_weight=",
                                    "resize="
                                    "temporal_loss=",
                                    "temporal_weight=",
                                    "iters=",
                                    "target_start=",
                                    "start_from_prev=",
                                    "style_from_prev=",
                                    "optimizer=",
                                    "early_stopping="])
    except:
        print(arg_help)
        sys.exit(2)

    content = None
    style = None
    # Optional arguments defaults
    output = None
    content_weight = 1
    style_weight = 1e5
    temporal_loss = True
    temporal_weight = 1e-3
    resize = 512
    target_start = 'content'
    start_from_prev = True  # If true, start optimizing from previous frame warped.
    iters = 300
    optimizer = 'lbfgs'
    style_from_prev = False  # If true, use previous (unwarped) stylized frame as style.
    early_stopping = False
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)
            sys.exit(2)
        elif opt in ("-c", "--content"):
            content = arg
        elif opt in ("-s", "--style"):
            style = arg
        elif opt in ("-o", "--output"):
            output = arg
        elif opt in ("-w", "--content_weight"):
            content_weight = float(arg)
        elif opt in ("-y", "--style_weight"):
            style_weight = float(arg)
        elif opt in ("-z", "--resize"):
            resize = int(arg)
        elif opt in ("-T", "--temporal_loss"):
            temporal_loss = True if arg == 'true' else False
        elif opt in ("-W", "--temporal_weight"):
            temporal_weight = float(arg)
        elif opt in ("-I", "--target_start"):
            target_start = arg
        elif opt in ("-p", "--start_from_prev"):
            start_from_prev = True if arg == 'true' else False
        elif opt in ("-Y", "--style_from_prev"):
            style_from_prev = True if arg == 'true' else False
        elif opt in ("-i", "--iters"):
            iters = int(arg)
        elif opt in ("-a", "--optimizer"):
            optimizer = arg
        elif opt in ("-e", "--early_stopping"):
            early_stopping = True if arg == 'true' else False

    if content is None or style is None:
        print(arg_help)
        sys.exit(2)

    if output is None:
        output = content.split('.')[0] + "_" + style + "." + content.split('.')[1]
    
    content = 'content/{}'.format(content)
    style = 'style/{}.jpg'.format(style)
    output = 'result/{}'.format(output)

    apply_style_transfer(content, style, output, content_weight,
                         style_weight, resize, temporal_loss, temporal_weight, 
                         target_start, start_from_prev, style_from_prev, 
                         optimizer, iters, early_stopping)


if __name__ == "__main__":
    main(sys.argv)
