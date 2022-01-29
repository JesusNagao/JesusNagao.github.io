clear all;
clc;


A=randn();
x=[0:0.1:10];
y=[0:0.1:10];
t=[1:length(x)];
n=10;

for i=1:n
    kx(i)=randn();
    ky(i)=randn();
    w(i)=randn();
end
[X,Y] = meshgrid(x,y);

f=A*sin(X*kx(1)+Y*ky(1));

for i=2:n
   
    f = f + A*sin(X*kx(i)+Y*ky(i));
    
end    

figure;
axis manual;
s = surf(X,Y,f);


for i=1:length(t)*5
    
    for j=2:n
        
        f = f + A*sin(X*kx(j)+Y*ky(j)-w(j)*i);
    
    end    
    
        s.XData=X;
        s.YData=Y;
        s.ZData=f;
    
        pause(0.05)
end
