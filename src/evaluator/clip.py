import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from PIL import Image

from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.functional.multimodal import clip_score
from functools import partial


class CLIPEvaluator(nn.Module):
    def __init__(
        self, 
        repo_id='openai/clip-vit-base-patch16',
        **kwargs
    ):
        super(CLIPEvaluator, self).__init__()
        self.repo_id = repo_id

        # create model and load pretrained weights from huggingface
        # self.clip = partial(clip_score, model_name_or_path=repo_id)
        self.clip = CLIPScore(model_name_or_path=repo_id)
        self.clip.reset()
    
    def forward(self, image, text):
        image, text = self.processing(image, text)
        image_int = (image * 255).to(torch.uint8)
        score = self.clip(image_int, text)
        score = score.item() / 100
        return score

    def processing(self, image, text):
        bsz, C, H, W = image.shape
        # import pdb; pdb.set_trace()
        if H != W:
            image = image.unfold(dimension=3, size=H, step=H//8) # cut image into several HxH images
            image = image.permute(0, 3, 1, 2, 4).squeeze(0)
            text = [text] * image.shape[0]
        return image, text
    
if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Load image
    # im_path = "logs/soundify-denoise/colorization/bell_example_29/img.png"
    # text = "a castle with bell towers, grayscale, lithograph style"
    im_path = "/home/czyang/Workspace/images-that-sound/logs/soundify-denoise/colorization/bell_example_29/img.png"
    text = "a castle with bell towers, grayscale, lithograph style"

    im = Image.open(im_path)
    im = TF.to_tensor(im).to(device)
    im = im.unsqueeze(0)
    # import pdb; pdb.set_trace()
    clip = CLIPEvaluator().to(device)
    score = clip(im, text)
    print(score)

    score = clip(im, text)
    print(score)



