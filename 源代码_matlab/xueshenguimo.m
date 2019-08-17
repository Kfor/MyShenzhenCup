%这个脚本用来预测缙云中学2022届学生人数
%原始数据
X=[2 3 4];
Y=[1250 1154 1197];
n=3;                %一共5个变量
 
x2=sum(X.^2);       % 求Σ(xi^2)
x1=sum(X);          % 求Σ(xi)
x1y1=sum(X.*Y);     % 求Σ(xi*yi)
y1=sum(Y);          % 求Σ(yi)
 
a=(n*x1y1-x1*y1)/(n*x2-x1*x1);      %解出直线斜率b=(y1-a*x1)/n
b=(y1-a*x1)/n;                      %解出直线截距
%作图
% 先把原始数据点用蓝色十字描出来
figure
plot(X,Y,'+');      
hold on
% 用红色绘制拟合出的直线
px=[1:0.1:6];
py=a*px+b;
plot(px,py,'r');
xlabel('年份');
ylabel('人数');
title('缙云中学学生规模预测');
hold on
plot(5,a*5+b,'*');
text(5,a*5+b,num2str(a*5+b));
