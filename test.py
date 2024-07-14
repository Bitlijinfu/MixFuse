import argparse
import os
from PIL import Image
from Network.MixFusion_Cross import MixFusion as M

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils.common import clamp, YCrCb2RGB
from data_utils.msrs_data import MSRS_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MixFusion')
    parser.add_argument('--dataset_name', default='MRI_SPECT', help='TNO MSRS Road Other MRI_PET')  
    parser.add_argument('--dataset_path', metavar='DIR', default='E:/fusion_dataset/Medicine/test', help='path to dataset') 
    parser.add_argument('--save_path', default='fusion_results')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--fusion_trained', default='models/MixFusion_Cross_add_224_Med.pth', help='use trained model')
    parser.add_argument('--cuda', default=True, type=bool, help='use GPU or not.')

    args = parser.parse_args()

    device = torch.device('cuda')
    input_path = os.path.join(args.dataset_path, args.dataset_name)

    test_dataset = MSRS_data(
        input_path, dataset_name=args.dataset_name, transform='test')

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    save_path = os.path.join(args.save_path, args.dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    model = M()

    window_size = 8
    model.load_state_dict(torch.load(args.fusion_trained, map_location='cpu'))
    model = model.to(device)
    model.eval()
    test_tqdm = tqdm(test_loader, total=len(test_loader))

    with torch.no_grad():
        for _, vis_y_image, cb, cr, inf_image, name in test_tqdm:
            vis_y_image = vis_y_image.to(device)
            cb = cb.to(device)
            cr = cr.to(device)
            inf_image = inf_image.to(device)

            _, _, h_old, w_old = vis_y_image.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            vis_y_image = torch.cat([vis_y_image, torch.flip(vis_y_image, [2])], 2)[
                :, :, :h_old + h_pad, :]
            vis_y_image = torch.cat([vis_y_image, torch.flip(vis_y_image, [3])], 3)[
                :, :, :, :w_old + w_pad]
            inf_image = torch.cat([inf_image, torch.flip(inf_image, [2])], 2)[
                :, :, :h_old + h_pad, :]
            inf_image = torch.cat([inf_image, torch.flip(inf_image, [3])], 3)[
                :, :, :, :w_old + w_pad]

            fused_image = model(vis_y_image, inf_image)

            fused_image = fused_image[..., :h_old, :w_old]
            fused_image = clamp(fused_image)
            fused_image = fused_image.detach().cpu().numpy()

            for k in range(len(name)):
                rgb_fused_image = fused_image[k]
                rgb_fused_image = rgb_fused_image.transpose((1, 2, 0))
                rgb_fused_image = (rgb_fused_image - np.min(rgb_fused_image)) / (
                    np.max(rgb_fused_image) - np.min(rgb_fused_image)
                )
                rgb_fused_image = np.uint8(255.0 * rgb_fused_image[:, :, 0])
                rgb_fused_image = Image.fromarray(rgb_fused_image)
                rgb_fused_image.save(f'{save_path}/{name[k]}')