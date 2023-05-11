 %Linear interpolation
long=size(X,1)
a=[1:1:long];
aa=linspace(1,long,360000) %在356828个数据点中取(360000)[根据保留时间决定]
AAA=[]
MS_long=365
for i=1:1:MS_long
    AAA(:,i)=interp1(a,X(:,i),aa);
end
