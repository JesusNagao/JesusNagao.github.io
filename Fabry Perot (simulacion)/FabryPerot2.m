clear all;
clc;

r=0.8; %Reflection coefficient
d=7.5e-6; %Distance between glass plates
epsilon=[0:1e-2:pi]; %Incidence angle
lambda=5e-6; %wavelength
w0=50; %Beam Waist

%[t,eps]=meshgrid(elevation,epsilon);

%Important relations
Delta=2*d*cos(epsilon);
delta=2*pi*Delta/lambda;
[d1,d2]=meshgrid(delta,delta);

F=(4*r^2)/((1-r^2)^2);
a=exp((-(d1).^2-(d2).^2)/w0); %Amplitude of the incident wave

I=a^2./(1+F.*(sin((d1.^2+d2.^2)./2).^2)); %Intensity of a Fabry-Perot interferometer


h=surf(I);
set(h,'LineStyle','none')
