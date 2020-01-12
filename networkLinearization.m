function [a,b] = networkLinearization(w10, w1, w20, w2, na, nb, tau, x0, fun)
%UNTITLED Summary of this function goes here
tan_deriv = @(x) (1 - tanh(x)^2);
a = zeros(na,1);
b = zeros(nb,1);
if strcmp(fun, 'tanh')
    for l=1:na
        for i = 1: length(w1)
            a(l) = a(l) - w2(i)*tan_deriv(w10 + w1*x0)*w1(i,l-tau+1);
        end
    end
    for l=1:nb
        if l < tau
           b(l) = 0; 
        else
           b(l) = b(1) + w2(i)*tan_deriv(w10 + w1*x0)*w1(i,l-tau+1);
        end
    end
end
end

