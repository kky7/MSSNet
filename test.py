import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from data.data_RGB import get_test_data
from skimage.metrics import peak_signal_noise_ratio
from model.MSSNet_eval import DeblurNet
import os
import argparse
import time

parser = argparse.ArgumentParser(description='DeblurNet test')
parser.add_argument("--test_datalist",type=str, default='./datalist/datalist_gopro_testset.txt')
parser.add_argument("--data_root_dir",type=str,default='./dataset')
parser.add_argument("--gpu",type=int, default=0)
parser.add_argument("--load_dir",type=str,default='./checkpoint/model_03000E.pt')
parser.add_argument("--outdir",type=str,default='./result/MSSNet')

parser.add_argument("--wf",type=int,default=54)
parser.add_argument("--scale",type=int,default=42)
parser.add_argument("--vscale",type=int,default=42)

parser.add_argument("--is_save", action="store_true")
parser.add_argument("--is_eval", action="store_true")

parser.set_defaults(is_save=False)
parser.set_defaults(is_eval=False)
args = parser.parse_args()

data_root_dir = args.data_root_dir
GPU = args.gpu

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
outdir = args.outdir
psnr_measure_dir = os.path.join(outdir,'measure')

if args.is_save:
    if not os.path.exists(outdir):
        os.makedirs(outdir)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = get_test_data(args.test_datalist,args.data_root_dir)
    dataloader = DataLoader(dataset=test_dataset, batch_size=1,num_workers=0,shuffle=False)
    torch.cuda.empty_cache()
    test_num = len(dataloader)
    print('test_num:',test_num)

    ##############################################################
    deblur_model = DeblurNet(wf=args.wf, scale=args.scale, vscale=args.vscale).cuda(GPU)

    if os.path.exists(args.load_dir):
        checkpoint = torch.load(str(args.load_dir))
        deblur_model.load_state_dict(checkpoint['model_state_dict'])
        steps = checkpoint['all_step']
        print('iterations:',steps)
    else:
        print("Checkpoint doesn't exist")
        raise ValueError

    deblur_model.eval()
    with torch.no_grad():
        # Warming up
        for iter_idx, data in enumerate(dataloader):
            print('Warming up %d iter'%(iter_idx))
            gt, blur, _ = data
            blur= blur.to(device)
            torch.cuda.synchronize()
            init_time = time.time()
            _ = deblur_model(blur)
            torch.cuda.synchronize()
            _ = time.time() - init_time

            if iter_idx == 10:
                break
        
        # Test
        itr_time = 0.0
        total_psnr = 0.0
        for iter_idx, data in enumerate(dataloader):
            gt, blur, blur_name = data

            blur= blur.to(device)

            torch.cuda.synchronize()
            init_time = time.time()
            out = deblur_model(blur)
            torch.cuda.synchronize()
            cur_time = time.time() - init_time
            itr_time += cur_time

            if args.is_eval:
                out_numpy = out.squeeze(0).cpu().numpy()
                gt_numpy = gt.squeeze(0).cpu().numpy()

                psnr = peak_signal_noise_ratio(out_numpy, gt_numpy, data_range=1)
                total_psnr += psnr

                print('%d iter PSNR: %.2f, time:%.3f' % (iter_idx + 1, psnr, cur_time))
            else:
                print('%d iter, time:%f'%(iter_idx+1,cur_time))

            if args.is_save:
                deblur_name = blur_name[0].replace('blur','deblur')
                save_dir = os.path.join(outdir, deblur_name)
                out += 0.5 / 255
                out = torch.clamp(out, 0, 1)
                out = F.to_pil_image(out.squeeze(0).cpu(), 'RGB')
                out.save(save_dir)


    print('Test Finish')
    print('==========================================================')
    if args.is_eval:
        print('The average PSNR: %.4f dB, Time: %.4f' % (total_psnr/test_num, itr_time/test_num))
    else:
        print('average test time: %f'%(itr_time/test_num))


if __name__ == '__main__':
    main()
