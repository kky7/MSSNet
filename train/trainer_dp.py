import torch
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio
from . import utils
from . import loss_L1_fft
from data.data_RGB import get_validation_data
from tqdm import tqdm

class Trainer():
    def __init__(self,args):
        self.arg=args
        self.max_epoch= args.max_epoch
        self.mgpu = args.mgpu
        self.data_root_dir = args.data_root_dir
        self.l1Loss = loss_L1_fft.L1Loss()
        self.fftLoss = loss_L1_fft.FFTLoss()
        self.checkdir = args.checkdir
        self.isloadch = args.isloadch
        self.isval = args.isval
        self.GPU =args.gpu

        if args.isval:
            val_dataset = get_validation_data(args.val_datalist,args.val_root_dir, {'patch_size':None})
            self.val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)

    def validation(self,deblur_model,train_writer,epoch):
        total_psnr = 0.
        val_num = len(self.val_loader)

        for data_val in tqdm(self.val_loader):
            deblur_model.eval()
            with torch.no_grad():
                gt_data = data_val[0]
                inp_data = data_val[1].to(self.GPU)

                out = deblur_model(inp_data)[-1].data
                out = torch.clamp(out,0,1)
                out_numpy = out.squeeze(0).cpu().numpy()
                gt_numpy = gt_data.squeeze(0).cpu().numpy()

                psnr = peak_signal_noise_ratio(out_numpy,gt_numpy,data_range=1)

                total_psnr += psnr


        mean_psnr = total_psnr / val_num
        print('mean psnr:',mean_psnr)

        train_writer.add_scalar('val_psnr', mean_psnr, epoch)

        return mean_psnr

    def save_mgpu_ch(self,deblur_model,optim,epoch,all_step,name):
        torch.save({
            'model_state_dict':deblur_model.module.state_dict(),
            'optimizer_state_dict':optim.state_dict(),
            'epoch': epoch,
            'all_step': all_step,
        },str(self.checkdir+ "/%s_%05dE.pt"%(name,epoch)))

    def save_ch(self,deblur_model,optim,epoch,all_step,name):

        print('saving dp')
        torch.save({
            'model_state_dict':deblur_model.state_dict(),
            'optimizer_state_dict':optim.state_dict(),
            'epoch': epoch,
            'all_step': all_step,
        },str(self.checkdir+ "/%s_%05dE.pt"%(name,epoch)))


    def train(self,deblur_model,train_dataloader,optim,scheduler,train_writer,start_epoch,all_step):
        train_batch_num = len(train_dataloader)

        for epoch in range(start_epoch,self.max_epoch+1):
            epoch_loss = 0
            deblur_model.train()
            start = 0
            for iteration, data in enumerate(train_dataloader):
                # zero_grad #########################
                for param in deblur_model.parameters():
                    param.grad = None
                #####################################

                all_step+=1

                gt = data[0].to(self.GPU)
                blur_images = data[1].to(self.GPU)

                output_module = deblur_model(blur_images)
                gt_pyramid = utils.get_pyramid(gt)
                gt_module = [gt_pyramid[1],gt_pyramid[2],gt_pyramid[2],gt_pyramid[2],gt_pyramid[2],gt_pyramid[2]]
                del gt_pyramid

                loss_l1 = np.sum([self.l1Loss(output_module[j],gt_module[j]) for j in range(len(output_module))])
                loss_fft = np.sum([self.fftLoss(output_module[j],gt_module[j]) for j in range(len(output_module))])
                loss = (loss_l1) + (0.1*loss_fft)

                loss.backward()
                optim.step()

                epoch_loss += loss.item()

                train_writer.add_scalar('epoch_loss',epoch_loss/train_batch_num, epoch)
                train_writer.add_scalar('lr',optim.param_groups[0]['lr'], epoch)

                if (iteration+1)%10 == 0:
                    stop = time.time()
                    print("epoch:%d /"%(epoch),"iter:%d /"%(all_step), "loss:%.4f /"%loss.item(),
                    '(%.3f s/100itr)'%(stop-start))
                    start = time.time()

                if all_step == 1:
                    if self.isval:
                        self.validation(deblur_model,train_writer,0)
                    train_writer.add_images('blur', utils.im2uint8(blur_images),0)
                    train_writer.add_images('s3_deblur', utils.gim2uint8(output_module[-1]),0)

                    print('save first iter checkpoint')
                    if self.mgpu:
                        self.save_mgpu_ch(deblur_model,optim,0,all_step,'model')
                    else:
                        self.save_ch(deblur_model,optim,0,all_step,'model')

            scheduler.step()

            train_writer.add_images('blur', utils.im2uint8(blur_images),epoch)
            train_writer.add_images('s3_deblur', utils.gim2uint8(output_module[-1]),epoch)

            if self.isval:
                if epoch==1 or epoch%600 == 0 or epoch == self.max_epoch:
                    _ = self.validation(deblur_model,train_writer,epoch)

            #Saving..################################################################
            if epoch==1 or epoch%600 == 0 or epoch == self.max_epoch:
                if self.mgpu:
                    self.save_mgpu_ch(deblur_model,optim,epoch,all_step,'model')
                else:
                    self.save_ch(deblur_model,optim,epoch,all_step,'model')
