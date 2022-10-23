close all;clear all;
%%
prompt = 'insert deblur file name: ';
prompt2 = 'insert gt file name: ';
prompt3 = 'test type:';
deblur_file_name = input(prompt,'s');
gt_file_name = input(prompt2,'s');
test_type = input(prompt3,'s');

eval_txt_name = strcat(test_type,'_matlab');
eval_txt_mean_name = strcat(test_type,'_matlab_mean');

file_path = strcat('./', deblur_file_name, '/');
gt_path = strcat('./', gt_file_name,'/');
path_list = [dir(strcat(file_path,'*.jpg')); dir(strcat(file_path,'*.png'))];
gt_list = [dir(strcat(gt_path,'*.jpg')); dir(strcat(gt_path,'*.png'))];
img_num = length(path_list);

fprintf('%d\n',img_num)
assert(img_num == length(gt_list))

evalFolderName = sprintf('%s',strcat( file_path,'evaluation'));                               
evalFolder     = fullfile(cd, evalFolderName);
if ~exist(evalFolder, 'dir')
    mkdir(evalFolder);
end

eval_txt = fopen(strcat(evalFolderName,'/',eval_txt_name,'.txt'),'W');
eval_txt_mean = fopen(strcat(evalFolderName,'/',eval_txt_mean_name,'.txt'),'W');

fprintf(eval_txt,strcat(deblur_file_name,'\n'));
fprintf(eval_txt_mean,strcat(deblur_file_name,'\n'));
%%
total_psnr = 0;
total_ssim = 0;
cnt = 0;

if img_num > 0 
    for j = 1:img_num
       cnt= cnt +1;
       image_name = path_list(j).name;
       gt_name = gt_list(j).name;
       input = imread(strcat(file_path,image_name));
       gt = imread(strcat(gt_path, gt_name));
       ssim_val = ssim(input, gt);
       psnr_val = psnr(input, gt);
       total_ssim = total_ssim + ssim_val;
       total_psnr = total_psnr + psnr_val;
       fprintf('%04dth:%s, gt:%s, psnr=%.3f, ssim=%.3f\n',j,image_name, gt_name, psnr_val, ssim_val)
       fprintf(eval_txt,'%s psnr:%.5f, ssim:%.5f\n',image_name, psnr_val,ssim_val);
   end
end
qm_psnr = total_psnr / img_num;
qm_ssim = total_ssim / img_num;

fprintf('%s PSNR: %f SSIM: %f\n', deblur_file_name , qm_psnr, qm_ssim);
fprintf(eval_txt_mean,'PSNR: %.4f\n',qm_psnr);
fprintf(eval_txt_mean,'SSIM: %.4f\n',qm_ssim);
fprintf(eval_txt_mean,'Test Num:%d\n',cnt);

fclose(eval_txt);
fclose(eval_txt_mean);
fprintf('%d\n',img_num);