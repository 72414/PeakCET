function [X,Xmz,tic,rt]=new_cdftransform(rt_first,rt_end)

% Input:
% rt_first为需要转化数据的保留时间起点
% rt_end为需要转化数据的保留时间终点
% Output:
% X为将cdf格式转成的mat格式，（rt2-rt1）保留时间段（单位为min）的矩阵形式
% tic为数据所有时间段的总流离子图
% rt为所有保留时间，rt_first与rt_end的设置需要在此时间范围内
% figure1为所有保留时间段的TIC图；figure2为（rt2-rt1）时间段的色谱图

fnc='03WenPJH202205-#2-0.4_400.cdf';                        %导入CDF文件

finfo = ncinfo(fnc);
rt=ncread(fnc,'scan_acquisition_time')/60;
tic=ncread(fnc,'total_intensity');
scan_index=ncread(fnc,'scan_index');
mass_values=ncread(fnc,'mass_values');                 % mass_values为所有的质荷比
intensity_values=ncread(fnc,'intensity_values');       % intensity_values为所有质荷比的离子强度
mz_min=34.9;              %选择需要转化的质荷比起点
mz_max=399.7;              %选择需要转化的质荷比终点
%mz_min=floor(mz_min);

figure;
plot(rt,tic)                  
idx1=find(rt>1,1,'first');  %起止坐标设置
idx2=find(rt>61,1,'first');   %返回大于61的第一个非零元素的位置
a=floor(mz_max-mz_min);  %计算floor(399.8-35.2)=365
mzmax=mz_min+a;

X=zeros(idx2-idx1+1,mzmax-mz_min+1);  %0矩阵
Xmz=X;

for i = idx1:1:idx2
    mz=mass_values(scan_index(i)+1:scan_index(i+1));    %得到mz，每个质荷比
    amz=scan_index(i+1)-scan_index(i);
    val=intensity_values(scan_index(i)+1:scan_index(i+1));    %mz对应的强度
    val_num=val;
    mz_num=floor(mz-mz_min+1);
    for j=2:size(mz_num,1)
        if mz_num(j)==mz_num(j-1)
            val_num(j)=val_num(j)+val_num(j-1); %质荷比很相近的把强度叠加在一起
            val_num(j-1)=0;
        end
    end
    Xmz(i-idx1+1,1:amz)=mz;
    X(i-idx1+1,mz_num)=val_num;                       %赋值给X
end
figure;
%%
mz_m=ncread(fnc,'mass_range_min');              %选择需要转化的质荷比起点
mz_x=ncread(fnc,'mass_range_max');  
plot(X)
%%

