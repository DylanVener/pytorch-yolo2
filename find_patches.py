import os
import sys
import time
import torch
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import do_detect, plot_boxes, find_bounds, load_class_names
from darknet import Darknet

def detect(m, imgfile, savepath = None):
    use_cuda = 1
    m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))
    
    boxes, layers = do_detect(m, sized, 0.5, 0.4, use_cuda)

    act = layers[29].cpu().data
    bounds = find_bounds(act, boxes)
    
    if len(bounds) == 0:
        return

    x1, x2, y1, y2 = max(bounds, key = lambda x : (x[1] - x[0])*(x[3] - x[2]))
    sect = act[:,:,int(round(x1)):int(round(x2)),int(round(y1)):int(round(y2))]

    print(sect.size())
    if savepath:
        torch.save(sect, savepath) 
    else:
        return sect
    

if __name__ == '__main__':
    cfgfile = sys.argv[1]
    weightfile = sys.argv[2]
    imgfolder = sys.argv[3]
    savefolder = sys.argv[4]

    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)

    for filename in os.listdir(imgfolder):
        if filename.endswith('.jpg'):
            detect(m, imgfolder + '/' + filename, savefolder + '/' + filename[:-4] + '.pt')
