function [J] = trajectoryFunction(u, y0, u0, k, N, Nu, dk, y_zad, lambda,w20,w2,w10,w1,na,nb,tau)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    u0 = [u0 u' u(end)*ones(1,N-Nu)];
    for j=1:N
        y0(k+j) = dk + w20 + w2*tanh(w10 + w1*[flip(u0(k-nb+j:k-tau+j)) flip(y0(k-na+j:k-1+j))]');
    end

    y0 = y0(k+1:k+N)';
    
    du(1) = u(1)-u0(k-1);
    for i=2:Nu
       du(i) = u(i) - u(i-1); 
    end
    J = sum((y_zad(k)*ones(N,1) - y0).^2) + lambda * sum(du.^2);
end

