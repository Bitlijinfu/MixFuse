clc
clear all

% easy = 1 means that the test runs are not time-consuming metrics
% easy = 0 means that the test is time-consuming metrics
easy = 1;
row_name1 = 'row1';
row_data1 = 'row2';

row = 'A';
row_name = strrep(row_name1, 'row', row);
row_data = strrep(row_data1, 'row', row);

% Medical image fusion methods
% medical_method = ["IFCNN", "SDNet", "U2Fusion", "SwinFusion", "MATR", "CDDFuse", "DAFusion"];
% medical_data   = ["MRI_CT", "MRI_PET", "MRI_PET_ours", "MRI_SPECT"];

% Define the fusion methods and datasets
vif_method = ["MixFuse"];
vif_data = ["MRI_PET","MRI_SPECT"];
% vif_data = ["MSRS", "TNO_21", "RoadScene"];

% Loop through each dataset
for i = 1:length(vif_data)
    part = strsplit(vif_data(i), '_');
    vi_name = part{1};        % MRI
    ir_name  = part{2};       % CT/PET/SPECT
%     vi_name = "vis";
%     ir_name = "ir";
    % Define the folder path for images
    fileFolder=fullfile('E:/fusion_dataset/Medicine/test', vif_data(i), vi_name);
    % Get the list of image files in the folder
    if i == 3
        dirOutput=dir(fullfile(fileFolder, '*.jpg'));
    else       
        dirOutput=dir(fullfile(fileFolder, '*.png'));
    end
    fileNames = {dirOutput.name};
    [m, num] = size(fileNames);

    for ii = 1:length(vif_method)
        % 遍历以方法为名字的融合结果文件夹
        method = vif_method(ii);
        Fused_dir = fullfile('C:/Users\LJF1328/Desktop/MixFuse/fusion_results', vif_data(i));
%         Fused_dir = fullfile('C:/Users\LJF1328/Desktop/MixFuse/fusion_results', method, vif_data(i));
        vi_dir = fullfile('E:/fusion_dataset/Medicine/test', vif_data(i), vi_name);
        ir_dir = fullfile('E:/fusion_dataset/Medicine/test', vif_data(i), ir_name);
%         'H:\DAFusion_result'+medical_method[index]+'results'+'MRI_CT or MRI_PET or MRI_SPECT'
%         Fused_dir = fullfile('E:/fusion_result/', method, vif_data(i));
%         Fused_dir = fullfile('E:/fusion_result/EDGEresult', vif_data(i));
%         vi_dir = fullfile('E:/fusion_dataset/VIS-IR/test', vif_data(i), vi_name);
%         ir_dir = fullfile('E:/fusion_dataset/VIS-IR/test', vif_data(i), ir_name);
        %  清空一次数据
        EN_set = [];    SF_set = [];    SD_set = [];    PSNR_set = [];
        MSE_set = [];   MI_set = [];    VIF_set = [];   AG_set = [];
        CC_set = [];    SCD_set = [];   Qabf_set = [];
        SSIM_set = [];      MS_SSIM_set = [];
        Nabf_set = [];      FMI_pixel_set = [];
        FMI_dct_set = [];   FMI_w_set = [];
        
        % 计算每张图像的指标
        for j = 1:num
            fileName_source_ir = fullfile(ir_dir, fileNames{j});
            fileName_source_vi = fullfile(vi_dir, fileNames{j}); 
            fileName_Fusion = fullfile(Fused_dir, fileNames{j});
            ir_image = imread(fileName_source_ir);
            vi_image = imread(fileName_source_vi);
            fused_image   = imread(fileName_Fusion);
            if size(ir_image, 3)>2
                ir_image = rgb2gray(ir_image);
            end

            if size(vi_image, 3)>2
                vi_image = rgb2gray(vi_image);
            end

            if size(fused_image, 3)>2
                fused_image = rgb2gray(fused_image);
            end
            [m, n] = size(fused_image);
            ir_size = size(ir_image);
            vi_size = size(vi_image);
            fusion_size = size(fused_image);
            if length(ir_size) < 3 & length(vi_size) < 3
                [EN, SF, SD, PSNR, MSE, MI, VIF, AG, CC, SCD, Qabf, Nabf, SSIM, MS_SSIM, FMI_pixel, FMI_dct, FMI_w] = analysis_Reference(fused_image,ir_image,vi_image, easy);
                EN_set = [EN_set, EN];
                SF_set = [SF_set, SF];
                SD_set = [SD_set, SD];
                MI_set = [MI_set, MI];
                AG_set = [AG_set, AG];
                Qabf_set = [Qabf_set, Qabf];
                VIF_set = [VIF_set, VIF];
                SCD_set = [SCD_set, SCD];
                SSIM_set = [SSIM_set, SSIM];
                CC_set = [CC_set, CC];
                PSNR_set = [PSNR_set, PSNR];
                MSE_set = [MSE_set, MSE];
                MS_SSIM_set = [MS_SSIM_set, MS_SSIM];
%                 Nabf_set = [Nabf_set, Nabf];
%                 FMI_pixel_set = [FMI_pixel_set, FMI_pixel];
%                 FMI_dct_set = [FMI_dct_set,FMI_dct];
%                 FMI_w_set = [FMI_w_set, FMI_w];
            else
                disp('unsucessful!')
                disp( fileName_Fusion)
            end
            fprintf('Datasets:%s, Fusion Method:%s, Image Index: %d, Image Name: %s\n', vif_data(i), method, j, fileNames{j})
        end
%         save_path = fullfile('E:/fusion_result/EDGEresult',vif_data(i));
        save_path = fullfile('C:/Users/LJF1328/Desktop/MixFuse/fusion_results');
        if ~exist(save_path, 'dir')
            mkdir(save_path);
        end
        file_name = fullfile(save_path, method + i + '_metric.xls') % File name for writing metrics to excel
        % 注意与数据一一对应
        metrics = [{'EN'},{'SF'},{'SD'},{'MI'},{'VIF'},{'Qabf'},{'AG'},{'SCD'},{'SSIM'},{'CC'},{'PSNR'},{'MSE'},{'MS_SSIM'}];
%         metrics = [{'EN'},{'SF'},{'VIF'},{'AG'},{'Qabf'},{'SSIM'}];

        if easy == 1
            xlswrite(file_name, metrics, method, 'B1')
            xlswrite(file_name, fileNames',method, 'A2')
            xlswrite(file_name, EN_set',method, 'B2')
            xlswrite(file_name, SF_set',method, 'C2')
            xlswrite(file_name, SD_set',method, 'D2') 
            xlswrite(file_name, MI_set',method, 'E2')
            xlswrite(file_name, VIF_set',method, 'F2')
            xlswrite(file_name, Qabf_set',method,'G2')
            xlswrite(file_name, AG_set',method, 'H2')
            xlswrite(file_name, SCD_set',method,'I2')
            xlswrite(file_name, SSIM_set',method,'J2')
            xlswrite(file_name, CC_set',method,'K2')
            xlswrite(file_name, PSNR_set',method,'L2')
            xlswrite(file_name, MSE_set',method,'M2')
             xlswrite(file_name, MSE_set',method,'N2')
        end
%         if easy ==1
%             xlswrite(file_name, {method},'EN',row_name)
%             xlswrite(file_name, {method},'SF',row_name)
%             xlswrite(file_name, {method},'SD',row_name) 
%             xlswrite(file_name, {method},'PSNR',row_name)
%             xlswrite(file_name, {method},'MSE',row_name)
%             xlswrite(file_name, {method},'MI',row_name)
%             xlswrite(file_name, {method},'VIF',row_name)
%             xlswrite(file_name, {method},'AG',row_name)
%             xlswrite(file_name, {method},'CC',row_name)
%             xlswrite(file_name, {method},'SCD',row_name)
%             xlswrite(file_name, {method},'Qabf',row_name)
% 
%             xlswrite(file_name,SF_set','SF',row_data)     
%             xlswrite(file_name,SD_set','SD',row_data) 
%             xlswrite(file_name,PSNR_set','PSNR',row_data)
%             xlswrite(file_name,MSE_set','MSE',row_data)
%             xlswrite(file_name,MI_set','MI',row_data)
%             xlswrite(file_name,VIF_set','VIF',row_data)
%             xlswrite(file_name,AG_set','AG',row_data)
%             xlswrite(file_name,CC_set','CC',row_data)
%             xlswrite(file_name,EN_set','EN',row_data)
%             xlswrite(file_name,SCD_set','SCD',row_data)
%             xlswrite(file_name,Qabf_set','Qabf',row_data)
%         else        
%             xlswrite(file_name, method,'Nabf',row_name)
%             xlswrite(file_name, method,'SSIM',row_name)
%             xlswrite(file_name, method,'MS_SSIM',row_name)
%             xlswrite(file_name, method,'FMI_pixel',row_name)
%             xlswrite(file_name, method,'FMI_dct',row_name)
%             xlswrite(file_name, method,'FMI_w',row_name)
%         
%             xlswrite(file_name,Nabf_set','Nabf',row_data)
%             xlswrite(file_name,SSIM_set','SSIM',row_data)
%             xlswrite(file_name,MS_SSIM_set','MS_SSIM',row_data)
%             xlswrite(file_name,FMI_pixel_set','FMI_pixel',row_data)
%             xlswrite(file_name,FMI_dct_set','FMI_dct',row_data)
%             xlswrite(file_name,FMI_w_set','FMI_w',row_data)
%         end
    end
end
% ir_dir = 'H:\DAFusion_result\PET\results\MRI_PET_RGB'; % Folder where infrared images are located
% vi_dir = 'H:\DAFusion_result\MRI\results\MRI_PET_RGB'; % Folder where visible images are located
% Fused_dir = 'H:\test_result\MATR\testMRI_PET'; % Folder where fused images are located
% 
% for j = 1:num
%     fileName_source_ir = fullfile(ir_dir, fileNames{j});
%     fileName_source_vi = fullfile(vi_dir, fileNames{j}); 
%     fileName_Fusion = fullfile(Fused_dir, fileNames{j});
%     ir_image = imread(fileName_source_ir);
%     vi_image = imread(fileName_source_vi);
%     fused_image   = imread(fileName_Fusion);
%     if size(ir_image, 3)>2
%         ir_image = rgb2gray(ir_image);
%     end
% 
%     if size(vi_image, 3)>2
%         vi_image = rgb2gray(vi_image);
%     end
% 
%     if size(fused_image, 3)>2
%         fused_image = rgb2gray(fused_image);
%     end
%     [m, n] = size(fused_image);
% %     fused_image = fused_image(7:m-6, 7:n-6);
%     ir_size = size(ir_image);
%     vi_size = size(vi_image);
%     fusion_size = size(fused_image);
%     if length(ir_size) < 3 & length(vi_size) < 3
%         [EN, SF,SD,PSNR,MSE, MI, VIF, AG, CC, SCD, Qabf, Nabf, SSIM, MS_SSIM, FMI_pixel, FMI_dct, FMI_w] = analysis_Reference(fused_image,ir_image,vi_image, easy);
%         EN_set = [EN_set, EN];SF_set = [SF_set,SF];SD_set = [SD_set, SD];PSNR_set = [PSNR_set, PSNR];
%         MSE_set = [MSE_set, MSE];MI_set = [MI_set, MI]; VIF_set = [VIF_set, VIF];
%         AG_set = [AG_set, AG]; CC_set = [CC_set, CC];SCD_set = [SCD_set, SCD];
%         Qabf_set = [Qabf_set, Qabf]; Nabf_set = [Nabf_set, Nabf];
%         SSIM_set = [SSIM_set, SSIM]; MS_SSIM_set = [MS_SSIM_set, MS_SSIM];
%         FMI_pixel_set = [FMI_pixel_set, FMI_pixel]; FMI_dct_set = [FMI_dct_set,FMI_dct];
%         FMI_w_set = [FMI_w_set, FMI_w];
%     else
%         disp('unsucessful!')
%         disp( fileName_Fusion)
%     end
%     fprintf('Fusion Method:%s, Image Name: %s\n', cell2mat(method_name), fileNames{j})
% end
% file_name = './metric.xls' % File name for writing metrics to excel
% if easy ==1
%     xlswrite(file_name, method_name,'EN',row_name)
%     xlswrite(file_name, method_name,'SF',row_name)
%     xlswrite(file_name, method_name,'SD',row_name) 
%     xlswrite(file_name, method_name,'PSNR',row_name)
%     xlswrite(file_name, method_name,'MSE',row_name)
%     xlswrite(file_name, method_name,'MI',row_name)
%     xlswrite(file_name, method_name,'VIF',row_name)
%     xlswrite(file_name, method_name,'AG',row_name)
%     xlswrite(file_name, method_name,'CC',row_name)
%     xlswrite(file_name, method_name,'SCD',row_name)
%     xlswrite(file_name, method_name,'Qabf',row_name)

%     xlswrite(file_name,SF_set','SF',row_data)
%     xlswrite(file_name,SD_set','SD',row_data) 
%     xlswrite(file_name,PSNR_set','PSNR',row_data)
%     xlswrite(file_name,MSE_set','MSE',row_data)
%     xlswrite(file_name,MI_set','MI',row_data)
%     xlswrite(file_name,VIF_set','VIF',row_data)
%     xlswrite(file_name,AG_set','AG',row_data)
%     xlswrite(file_name,CC_set','CC',row_data)
%     xlswrite(file_name,EN_set','EN',row_data)
%     xlswrite(file_name,SCD_set','SCD',row_data)
%     xlswrite(file_name,Qabf_set','Qabf',row_data)
% else        
%     xlswrite(file_name, method_name,'Nabf',row_name)
%     xlswrite(file_name, method_name,'SSIM',row_name)
%     xlswrite(file_name, method_name,'MS_SSIM',row_name)
%     xlswrite(file_name, method_name,'FMI_pixel',row_name)
%     xlswrite(file_name, method_name,'FMI_dct',row_name)
%     xlswrite(file_name, method_name,'FMI_w',row_name)
% 
%     xlswrite(file_name,Nabf_set','Nabf',row_data)
%     xlswrite(file_name,SSIM_set','SSIM',row_data)
%     xlswrite(file_name,MS_SSIM_set','MS_SSIM',row_data)
%     xlswrite(file_name,FMI_pixel_set','FMI_pixel',row_data)
%     xlswrite(file_name,FMI_dct_set','FMI_dct',row_data)
%     xlswrite(file_name,FMI_w_set','FMI_w',row_data)
% end
