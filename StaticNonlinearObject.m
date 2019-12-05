function [y] = StaticNonlinearObject(u)
%STATICMODEL Summary of this function goes here
%   Detailed explanation goes here
global alfa1;
global alfa2;
global beta1;
global beta2;

global g1;
global g2;

y = g2(((beta1+ beta2)*g1(u))/(1+alfa1+alfa2));
end

