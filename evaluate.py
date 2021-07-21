from PIL import Image


import yaml
from models.detr import build_evaluate
from util.eval_util import detect, plot_results
import torch
import torchvision.transforms as T
torch.set_grad_enabled(False)


with open('config/cfg.yaml', 'r') as loadfile:
    cfg = yaml.load_all(loadfile)
    cfg_all = [x for x in cfg]

# evaluate mode
cfg = cfg_all[1]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize((cfg['scaled_width'], cfg['scaled_height'])),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# build model
detr = build_evaluate(cfg)
detr.load_state_dict(torch.load(
    '{}/wt.pt'.format(cfg['output_dir']), map_location='cpu'))
detr.eval()

url = '../train/000300.jpg'
im = Image.open(url)

scores, boxes = detect(im, detr, transform)

plot_results(im, scores, boxes, cfg['classes'], cfg['colors'])
