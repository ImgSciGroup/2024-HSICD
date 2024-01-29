
clear all

addpath newfusion
addpath nsst_toolbox
addpath Dataset
addpath 图像融合评价标准13项指标
% image sets
names = {'Camp', 'Camp1', 'Dune', 'Gun', 'Navi', 'Kayak', 'Octec', 'Road', 'Road2' 'Steamboat', 'T2', 'T3', 'Trees4906', 'Trees4917','Car','G8','G9','G10','Plane','P21','H6','H12','H13','H16','H18','T7','T19','T17'};
save_F = zeros(463,241);
% read one image set
setName = names{23};
% 

IB=load('D:/text_all/test_river/t1_25_t2_0_100_zsy587632_new.mat').HypeRvieW(:,:,2);
IA=load('D:/text_all/test_river/t1_25_t2_0_100_zsy587632_new.mat').HypeRvieW(:,:,1);
IC=load('D:/text_all/test_river/t1_25_t2_0_100_zsy587632_new.mat').HypeRvieW(:,:,3);
IA=(IA-min(min(IA)))/(max(max(IA))-min(min(IA)))*255;
IB=(IB-min(min(IB)))/(max(max(IB))-min(min(IB)))*255;
IC=(IC-min(min(IC)))/(max(max(IC))-min(min(IC)))*255;
[p,q]=size(IA);
A=IA;
B=IB;
C=IC;
%% NSST decomposition
pfilt = 'maxflat';
shear_parameters.dcomp =[5,5,3,3];
shear_parameters.dsize =[16,16,8,8];
[y1,shear_f1]=nsst_dec2(A,shear_parameters,pfilt);
[y2,shear_f2]=nsst_dec2(B,shear_parameters,pfilt);
[y3,shear_f3]=nsst_dec2(C,shear_parameters,pfilt);
%% fusion
y=y1;
y{1}=(y1{1}+y2{1}+y3{1})*0.33
for m=2:length(shear_parameters.dcomp)+1
    temp=size((y1{m}));
    temp=temp(3);
     for n=1:temp
        Ahigh=y1{m}(:,:,n);
        Bhigh=y2{m}(:,:,n);
        Chigh=y3{m}(:,:,n);
        max1 = max( Ahigh,Bhigh);
        y{m}(:,:,n) = max( max1,Chigh);
    end
end

% figure,subplot(2,2,1),imshow(Ahigh,[]);title('A高频');
% subplot(2,2,2),imshow(Bhigh,[]);title('B高频');
% subplot(2,2,3),imshow(Chigh,[]);title('C高频');
% subplot(2,2,4),imshow(Dhigh,[]);title('D高频');
% figure;imshow(y{m}(:,:,n),[]);title('AB高频融合');
%%  NSST reconstruction
F=nsst_rec2(y,shear_f1,pfilt);
F= abs(F);
final_CI= (F-min(min(F)))/(max(max(F))-min(min(F)));
final_CI=final_CI*255;
final_CI=uint8(final_CI);
F=uint8(imresize(F,[p q]));
%% save
%fileName = strcat('D:\Fuse\',setName,'_fuse_1.png');
%imwrite(F,fileName);