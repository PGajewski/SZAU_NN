function [y_vector,x_vector, u_vector] = NonlinearObject(u, x_0, iteration)
%NONLINEAROBJECT Simulate object for for discrete time
%   u - control signal
%   x_0 - init state vector
%   iteration - number of iterations.
global alfa1;
global alfa2;
global beta1;
global beta2;

global g1;
global g2;

if nargin > 2
  it = iteration;
else
  it = 1;
end
x = x_0;
x_vector = zeros(2,it);
y_vector = zeros(1,it);
u_vector = zeros(1,it);
for i=1 : it
    u_vector(1,i) = u;
    x_vector(1,i) = -alfa1 * x(1) + x(2) + beta1*g1(u);
    x_vector(2,i) = -alfa2 * x(1) + beta2*g1(u);
    y_vector(1,i) = g2(x(1));
    x = x_vector(:,i);
end
end

