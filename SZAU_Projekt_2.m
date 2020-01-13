%%Data
clear;
close all;

global alfa1;
global alfa2;
global beta1;
global beta2;

global g1;
global g2;
global u_min;
global u_max;
global u_delay;

alfa1 = -1.489028;
alfa2 = 0.535261;
beta1 = 0.012757;
beta2 = 0.010360;
u_min = -1;
u_max = 1;
u_delay = 5;

g1 = @(u)((exp(7*u)-1)/(exp(7*u)+1));
g2 = @(x)(0.25*(1-exp(-2.5*x)));

%% Static characteristic.
u = u_min:0.01:u_max;
y_stat = zeros(1, length(u));
for i =1:length(u)
   y_stat(i) = StaticNonlinearObject(u(i)); 
end

figure;
plot(u,y_stat);
title('Charakterystyka statyczna procesu');
xlabel('u');
ylabel('y_{stat}');

%% Generate data for neural network.
seeds = [16, 50];
files = {'dane.txt','dane_wer.txt'};
plot_texts = {'Symulacja procesu (dane ucz�ce)','Symulacja procesu (dane weryfikuj�ce)'};
file_template = '%f %f\n';

%k=1 - learning data
%k=2 - verifying data
for k=1:2
    rng(seeds(k));
    steps = 40;
    steps_length = 50;
    u = [(u_delay+1), 0]; %Init signal.
    %Generate random signals.
    for i = 2:steps
        u(i,:) = [(i*50) (u_min + rand*(u_max-u_min))];
    end

    y_vector = zeros(1,steps*steps_length);
    x_vector = zeros(2,steps*steps_length);
    u_vector = zeros(1,steps*steps_length);
    
    %Init state
    x_vector(:,1) = [0;0];
    
    %Main simulation loop.
    for i=1:length(u)-1
        for j=u(i,1) : u(i+1,1)-1
            u_vector(1,j) = u(i,2);
            x_vector(1,j+1) = -alfa1 * x_vector(1,j) + x_vector(2,j) + beta1*g1(u_vector(1,j-u_delay));
            x_vector(2,j+1) = -alfa2 * x_vector(1,j) + beta2*g1(u_vector(1,j-u_delay));
            %y_vector(1,j) = g2(x_vector(1,j)); %without noise
            y_vector(1,j) = g2(x_vector(1,j)) + 0.02*(rand(1,1)-0.5); % with noise
        end
    end

    %Display data.
%     figure;
%     subplot(2,1,1);
%     plot(1:length(y_vector),y_vector, 'r');
%     title(strcat(plot_texts{k}, ' - wyj�cie'));
%     xlabel('k');
%     ylabel('y');
%     subplot(2,1,2);
%     plot(1:length(u_vector),u_vector, 'b');
%     title(strcat(plot_texts{k}, ' - sterowanie'));
%     xlabel('k');
%     ylabel('u');
    figure;
    plot(1:length(y_vector),y_vector);
    hold on;
    title(plot_texts{k});
    plot(1:length(u_vector),u_vector);
    xlabel('k');
    ylabel('y,u');
    legend('y','u');
    
    %Save to file
    fileID = fopen(files{k},'w');
    for i=1:length(u_vector)
        fprintf(fileID,file_template,u_vector(i),y_vector(i));
    end
    fclose(fileID);
end
%% Delay
u=[zeros(1,5) ones(1,10)];
x1 = zeros(1,15);
x2 = zeros(1,15);
y = zeros(1,15);
for k = 6:15
    x1(k) = -alfa1*x1(k-1) + x2(k-1) + beta1*g1(u(k-5));
    x2(k) = -alfa2*x1(k-1) + beta2*g1(u(k-5));
    y(k) = g2(x1(k));
end
figure;
subplot(2,1,1);
plot(1:15, y);
title('Prezentacja opoznienia procesu - wyj�cie');
xlabel('k');
ylabel('y');
subplot(2,1,2);
plot(1:15, u);
title('Prezentacja opoznienia procesu - sterowanie');
xlabel('k');
ylabel('u');


save('beforemodels.mat'); % save data before models
%% Best model OE and BFGS, simul OE
% Read data.
[u_val, y_val] = readData('dane_wer.txt');
[u_learning, y_learning] = readData('dane.txt');

nb = 6;
na = 2;
tau = 5;

save('tmp.mat');
%Load model of network.
run('MODELOEBFGS/NajlepszyK=4model=7/model.m');
load('tmp.mat');
save('tmp.mat');

run('MODELOEBFGS/NajlepszyK=4model=7/uczenie.m');
load('tmp.mat');
delete('tmp.mat');

%Count response of network for learning data.
y_vector = zeros(1,length(u_val));
y_vector(1:nb) = y_learning(1:nb);

for i=(nb)+1:length(u_val)
    y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_learning(i-nb:i-tau)) flip(y_vector(i-na:i-1))]');
end

%errLearning =(y_vector-y_learning)*(y_vector-y_learning)'

%Compare model for learning data.
figure;
hold on;
plot(1:length(y_vector),y_vector);
plot(1:length(y_learning),y_learning);
plot(1:length(u_learning),u_learning);
hold off;
title('Symulacja OE modelu neuronowego dla danych ucz., tryb OE, algorytm BGFS');
legend('y_{learning}','y_{model}','u');
xlabel('t');
ylabel('y,u');

figure;
h = scatter(y_learning,y_vector,0.1);
h.Marker='.';
title('Relacja OE wyj�cia procesu i modelu, dane ucz., tryb OE, algorytm BFGS');
xlabel('y_{learning}');
ylabel('y_{mod}');

%Count response of network for validate data.
y_vector = zeros(1,length(u_val));
y_vector(1:nb) = y_val(1:nb);

for i=(nb)+1:length(u_val)
    y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_val(i-nb:i-tau)) flip(y_vector(i-na:i-1))]');
end

%errValidating =(y_vector-y_val)*(y_vector-y_val)'

%Compare model for validating data.
figure;
hold on;
plot(1:length(y_vector),y_vector);
plot(1:length(y_val),y_val);
plot(1:length(u_val),u_val);
hold off;
title('Symulacja OE modelu neuronowego dla danych wer., tryb OE, algorytm BFGS');
legend('y_{val}','y_{model}','u');
xlabel('t');
ylabel('y,u');

figure;
h = scatter(y_val,y_vector,0.1);
h.Marker='.';
title('Relacja OE wyj�cia procesu i modelu, dane wer., tryb OE, algorytm BFGS');
xlabel('y_{val}');
ylabel('y_{mod}');

save('bestmodel.mat');
clear;
load('beforemodels.mat');
%% Best model OE and SD, simul OE
% Read data.
[u_val, y_val] = readData('dane_wer.txt');
[u_learning, y_learning] = readData('dane.txt');

nb = 6;
na = 2;
tau = 5;

save('tmp.mat');
%Load model of network.
run('MODELOESD/Najlepszymodel=9/model.m');
load('tmp.mat');
save('tmp.mat');

run('MODELOESD/Najlepszymodel=9/uczenie.m');
load('tmp.mat');
delete('tmp.mat');

%Count response of network for learning data.
y_vector = zeros(1,length(u_val));
y_vector(1:nb) = y_learning(1:nb);

for i=(nb)+1:length(u_val)
    y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_learning(i-nb:i-tau)) flip(y_vector(i-na:i-1))]');
end

%errLearning =(y_vector-y_learning)*(y_vector-y_learning)'

%Compare model for learning data.
figure;
hold on;
plot(1:length(y_vector),y_vector);
plot(1:length(y_learning),y_learning);
plot(1:length(u_learning),u_learning);
hold off;
title('Symulacja OE modelu neuronowego dla danych ucz., tryb OE, algorytm SD');
legend('y_{learning}','y_{model}','u');
xlabel('t');
ylabel('y,u');

figure;
h = scatter(y_learning,y_vector,0.1);
h.Marker='.';
title('Relacja OE wyj�cia procesu i modelu, dane ucz., tryb OE, algorytm SD');
xlabel('y_{learning}');
ylabel('y_{mod}');

%Count response of network for validate data.
y_vector = zeros(1,length(u_val));
y_vector(1:nb) = y_val(1:nb);

for i=(nb)+1:length(u_val)
    y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_val(i-nb:i-tau)) flip(y_vector(i-na:i-1))]');
end

%errValidating =(y_vector-y_val)*(y_vector-y_val)'

%Compare model for validating data.
figure;
hold on;
plot(1:length(y_vector),y_vector);
plot(1:length(y_val),y_val);
plot(1:length(u_val),u_val);
hold off;
title('Symulacja OE modelu neuronowego dla danych wer., tryb OE, algorytm SD');
legend('y_{val}','y_{model}','u');
xlabel('t');
ylabel('y,u');

figure;
h = scatter(y_val,y_vector,0.1);
h.Marker='.';
title('Relacja OE wyj�cia procesu i modelu, dane wer., tryb OE, algorytm SD');
xlabel('y_{val}');
ylabel('y_{mod}');

clear;
load('beforemodels.mat');
%% Best model ARX and BFGS, simul ARX
% Read data.
[u_val, y_val] = readData('dane_wer.txt');
[u_learning, y_learning] = readData('dane.txt');

nb = 6;
na = 2;
tau = 5;

save('tmp.mat');
%Load model of network.
run('MODELARXBFGS/Najlepszymodel=1/model.m');
load('tmp.mat');
save('tmp.mat');

run('MODELARXBFGS/Najlepszymodel=1/uczenie.m');
load('tmp.mat');
delete('tmp.mat');

%Count response of network for learning data.
y_vector = zeros(1,length(u_val));
y_vector(1:nb) = y_learning(1:nb);

for i=(nb)+1:length(u_val)
    y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_learning(i-nb:i-tau)) flip(y_learning(i-na:i-1))]');
end

%errLearning =(y_vector-y_learning)*(y_vector-y_learning)'

%Compare model for learning data.
figure;
hold on;
plot(1:length(y_vector),y_vector);
plot(1:length(y_learning),y_learning);
plot(1:length(u_learning),u_learning);
hold off;
title('Symulacja ARX modelu neuronowego dla danych ucz., tryb ARX, algorytm BGFS');
legend('y_{learning}','y_{model}','u');
xlabel('t');
ylabel('y,u');

figure;
h = scatter(y_learning,y_vector,0.1);
h.Marker='.';
title('Relacja ARX wyj�cia procesu i modelu, dane ucz., tryb ARX, algorytm BFGS');
xlabel('y_{learning}');
ylabel('y_{mod}');

%Count response of network for validate data.
y_vector = zeros(1,length(u_val));
y_vector(1:nb) = y_val(1:nb);

for i=(nb)+1:length(u_val)
    y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_val(i-nb:i-tau)) flip(y_val(i-na:i-1))]');
end

%errValidating =(y_vector-y_val)*(y_vector-y_val)'

%Compare model for validating data.
figure;
hold on;
plot(1:length(y_vector),y_vector);
plot(1:length(y_val),y_val);
plot(1:length(u_val),u_val);
hold off;
title('Symulacja ARX modelu neuronowego dla danych wer., tryb ARX, algorytm BFGS');
legend('y_{val}','y_{model}','u');
xlabel('t');
ylabel('y,u');

figure;
h = scatter(y_val,y_vector,0.1);
h.Marker='.';
title('Relacja ARX wyj�cia procesu i modelu, dane wer.,tryb ARX, algorytm BFGS');
xlabel('y_{val}');
ylabel('y_{mod}');

clear;
load('beforemodels.mat');
%% Best model ARX and BFGS, simul OE
% Read data.
[u_val, y_val] = readData('dane_wer.txt');
[u_learning, y_learning] = readData('dane.txt');

nb = 6;
na = 2;
tau = 5;

save('tmp.mat');
%Load model of network.
run('MODELARXBFGS/Najlepszymodel=1/model.m');
load('tmp.mat');
save('tmp.mat');

run('MODELARXBFGS/Najlepszymodel=1/uczenie.m');
load('tmp.mat');
delete('tmp.mat');

%Count response of network for learning data.
y_vector = zeros(1,length(u_val));
y_vector(1:nb) = y_learning(1:nb);

for i=(nb)+1:length(u_val)
    y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_learning(i-nb:i-tau)) flip(y_vector(i-na:i-1))]');
end

%errLearning =(y_vector-y_learning)*(y_vector-y_learning)'

%Compare model for learning data.
figure;
hold on;
plot(1:length(y_vector),y_vector);
plot(1:length(y_learning),y_learning);
plot(1:length(u_learning),u_learning);
hold off;
title('Symulacja OE modelu neuronowego dla danych ucz., tryb ARX, algorytm BGFS');
legend('y_{learning}','y_{model}','u');
xlabel('t');
ylabel('y,u');

figure;
h = scatter(y_learning,y_vector,0.1);
h.Marker='.';
title('Relacja OE wyj�cia procesu i modelu, dane ucz., tryb ARX, algorytm BFGS');
xlabel('y_{learning}');
ylabel('y_{mod}');

%Count response of network for validate data.
y_vector = zeros(1,length(u_val));
y_vector(1:nb) = y_val(1:nb);

for i=(nb)+1:length(u_val)
    y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_val(i-nb:i-tau)) flip(y_vector(i-na:i-1))]');
end

%errValidating =(y_vector-y_val)*(y_vector-y_val)'

%Compare model for validating data.
figure;
hold on;
plot(1:length(y_vector),y_vector);
plot(1:length(y_val),y_val);
plot(1:length(u_val),u_val);
hold off;
title('Symulacja OE modelu neuronowego dla danych wer., tryb ARX, algorytm BFGS');
legend('y_{val}','y_{model}','u');
xlabel('t');
ylabel('y,u');

figure;
h = scatter(y_val,y_vector,0.1);
h.Marker='.';
title('Relacja OE wyj�cia procesu i modelu, dane wer., tryb ARX, algorytm BFGS');
xlabel('y_{val}');
ylabel('y_{mod}');


clear;
load('beforemodels.mat');
% %% Validate object
% sim_type = 'OE'; % ARMAX OE
% Read config.
[tau, nb, na, K, max_iter, error, algorithm] = readConfig();
% 
% %Compare model type and chosen simulation type.
% if strcmp(sim_type,'OE') &&  (algorithm==1)
%     warning('Chosen OE simulation, but model is ARMAX');
% elseif strcmp(sim_type,'ARX') &&  (algorithm==2)
%     warning('Chosen ARMAX simulation, but model is OE');
% end
% 
% % Read data.
% [u_val, y_val] = readData('dane_wer.txt');
% [u_learning, y_learning] = readData('dane.txt');
% 
% save('tmp.mat');
% %Load model of network.
% model;
% load('tmp.mat');
% save('tmp.mat');
% 
% uczenie
% load('tmp.mat');
% delete('tmp.mat');
% 
% %Count response of network for learning data.
% y_vector = zeros(1,length(u_val));
% y_vector(1:nb) = y_learning(1:nb);
% 
% for i=(nb)+1:length(u_val)
%     if strcmp(sim_type,'OE')
%         y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_learning(i-nb:i-tau)) flip(y_vector(i-na:i-1))]');
%     elseif strcmp(sim_type,'ARX')
%         y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_learning(i-nb:i-tau)) flip(y_learning(i-na:i-1))]');
%     end
% end
% 
% %Compare model for learning data.
% figure;
% hold on;
% plot(1:length(y_vector),y_vector);
% plot(1:length(y_learning),y_learning);
% plot(1:length(u_learning),u_learning);
% hold off;
% if strcmp(sim_type,'OE')
%     title('Symulacja modelu neuronowego dla danych ucz�cych w trybie OE');
% end
% if strcmp(sim_type,'ARX')
%     title('Symulacja modelu neuronowego dla danych ucz�cych w trybie ARX');
% end
% legend('y_{learning}','y_{model}','u');
% xlabel('t');
% ylabel('y,u');
% 
% figure;
% h = scatter(y_learning,y_vector,0.1);
% h.Marker='.';
% if strcmp(sim_type,'OE')
%     title('Relacja wyj�cia procesu i wyj�ciowego modelu, w trybie OE, dane ucz�ce');
% end
% if strcmp(sim_type,'ARX')
%     title('Relacja wyj�cia procesu i wyj�ciowego modelu, w trybie ARX, dane ucz�ce');
% end
% xlabel('y_{learning}');
% ylabel('y_{mod}');
% err =(y_vector-y_learning)*(y_vector-y_learning)';
% 
% %Count response of network for validate data.
% y_vector = zeros(1,length(u_val));
% y_vector(1:nb) = y_val(1:nb);
% 
% for i=(nb)+1:length(u_val)
%     if strcmp(sim_type,'OE')
%         y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_val(i-nb:i-tau)) flip(y_vector(i-na:i-1))]');
%     elseif strcmp(sim_type,'ARX')
%         y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_val(i-nb:i-tau)) flip(y_val(i-na:i-1))]');
%     end
% end
% 
% %Compare model for validating data.
% figure;
% hold on;
% plot(1:length(y_vector),y_vector);
% plot(1:length(y_val),y_val);
% plot(1:length(u_val),u_val);
% hold off;
% if strcmp(sim_type,'OE')
%     title('Symulacja modelu neuronowego dla danych weryfikuj�cych, w trybie OE');
% end
% if strcmp(sim_type,'ARX')
%     title('Symulacja modelu neuronowego dla danych weryfikuj�cych, w trybie ARX');
% end
% legend('y_{val}','y_{model}','u');
% xlabel('t');
% ylabel('y,u');
% 
% figure;
% h = scatter(y_val,y_vector,0.1);
% h.Marker='.';
% if strcmp(sim_type,'OE')
%     title('Relacja wyj�cia procesu i wyj�ciowego modelu, w trybie OE, dane weryfikuj�ce');
% end
% if strcmp(sim_type,'ARX')
%     title('Relacja wyj�cia procesu i wyj�ciowego modelu, w trybie ARX, dane weryfikuj�ce');
% end
% xlabel('y_{val}');
% ylabel('y_{mod}');
% 
% %Count MSE.
% %err = immse(y_vector,y_val)
% err =(y_vector-y_val)*(y_vector-y_val)';
%% Prepare mean square linear model.

% Read data
[u_val, y_val] = readData('dane_wer.txt');
[u_learning, y_learning] = readData('dane.txt');

nb = 6;
na = 2;
tau = 5;

%Construct M matrix.
M = zeros(length(y_learning)-nb,nb-tau+na+2);
for i=(nb+1):length(y_learning)
    M(i-nb,:) = [1 flip(u_learning(i-nb:i-tau)) flip(y_learning(i-na:i-1))];
end

W =M\y_learning(nb+1:end)';

y_vector_poly = zeros(1,length(u_val));
y_vector_poly(1:nb) = y_learning(1:nb);
for i=(nb)+1:length(u_val)
    y_vector_poly(1,i) = W'*[1 flip(u_learning(i-nb:i-tau)) flip(y_vector_poly(i-na:i-1))]';
end

%errLearning =(y_vector_poly-y_learning)*(y_vector_poly-y_learning)'

%Compare model for learning data.
figure;
hold on;
plot(1:length(y_vector_poly),y_vector_poly);
plot(1:length(y_learning),y_learning);
plot(1:length(u_learning),u_learning);
hold off;
title('Symulacja modelu MNK dla danych ucz�cych');
legend('y_{learning}','y_{model}','u');
xlabel('t');
ylabel('y,u');

figure;
h = scatter(y_learning,y_vector_poly,0.1);
h.Marker='.';
title('Relacja modelu MNK dla danych ucz�cych');
xlabel('y_{learning}');
ylabel('y_{mod}');


y_vector_poly = zeros(1,length(u_val));
y_vector_poly(1:nb) = y_val(1:nb);
for i=(nb)+1:length(u_val)
    y_vector_poly(1,i) = W'*[1 flip(u_val(i-nb:i-tau)) flip(y_vector_poly(i-na:i-1))]';
end

%errValidating =(y_vector_poly-y_val)*(y_vector_poly-y_val)'

%Compare model for validating data.
figure;
hold on;
plot(1:length(y_vector_poly),y_vector_poly);
plot(1:length(y_val),y_val);
plot(1:length(u_val),u_val);
hold off;
title('Symulacja modelu MNK dla danych weryfikuj�cych');
legend('y_{val}','y_{model}','u');
xlabel('t');
ylabel('y,u');

figure;
h = scatter(y_val,y_vector_poly,0.1);
h.Marker='.';
title('Relacja modelu MNK dla danych weryfikuj�cych');
xlabel('y_{val}');
ylabel('y_{mod}');

%Count MSE.
err = immse(y_vector_poly,y_val)

%% Process regulation.
alg_type = 'NO'; %NPL, GPC, PID, NO
[tau, nb, na, K, max_iter, error, algorithm] = readConfig();
load('bestmodel');


sim_time = 500;
T = 0.1;

lambda = 1;
N = 6;
Nu = 2;

K = 1.8;
Ti = 12*T;
Td = 3*T;

%Initialization.
y_vector = zeros(1,sim_time);
x_vector = zeros(2,sim_time);
u_vector = zeros(1,sim_time);

%Generate changing set values vector.
steps = [0,  0;
        100, 0.1;
        200, 0;
        300, 0.05;
        500, 0];
    
y_zad = zeros(sim_time,1);
for i=1:sim_time
    for j=2:length(steps)
        if steps(j,1) > i
            y_zad(i) = steps(j-1,2);
           break; 
        end
    end
end
switch alg_type
    case 'NPL'
    case 'GPC'
        u0 = 0.12;
        y0 = 0.0978;
        
        %Create linearization point vector.
        x = [ones(1,nb-tau+1)*u0 ones(1,na)*y0]';

        %Count linearization coefficients.
        [a,b] = networkLinearizationNum(w10, w1, w20, w2, na, nb, tau, x, 'tanh');
        %[a,b] = networkLinearization(w10, w1, w20, w2, na, nb, tau, x, 'tanh');

        %Count step response.
        s = zeros(N,1);
        for j=1:N
           for i=1:min(j,nb)
               s(j) = s(j) + b(i);
           end
           for i=1:min(j-1,na)
               s(j) = s(j) - a(i)*s(j-i);
           end
        end

        %Construct dynamic matrix.
        M = zeros(N,Nu);
        for j=1: Nu
           M(:,j) = [zeros(j-1,1); s(1:N-j+1)]';
        end

        K = (M' * M + lambda * eye(Nu))\(M');        
    case 'PID'
        e_vector = zeros(1,sim_time);
        r2 = K*Td/T;
        r1 = K*(T/(2*Ti)-2*Td/T-1);
        r0 = K*(1+T/(2*Ti)+Td/T);
    case 'NO'
        u_start = zeros(Nu,1);
        options = optimoptions('fmincon','Algorithm','sqp','StepTolerance',1e-12);
end

%Init state
x_vector(:,1) = [0;0];
%Main simulation loop.
for k=nb+1:sim_time
    k
    %Object simulation.
    x_vector(1,k+1) = -alfa1 * x_vector(1,k) + x_vector(2,k) + beta1*g1(u_vector(1,k-u_delay));
    x_vector(2,k+1) = -alfa2 * x_vector(1,k) + beta2*g1(u_vector(1,k-u_delay));
    y_vector(1,k) = g2(x_vector(1,k)); %without noise
    %y_vector(1,k) = g2(x_vector(1,k)) + 0.02*(rand(1,1)-0.5); % with noise
    %y_vector(1,k) = w20 + w2*tanh(w10 + w1*[flip(u_vector(k-nb:k-tau)) flip(y_vector(k-na:k-1))]');

    %Preprocesing
    switch alg_type
        case 'NPL'
            %Create linearization point vector.
            x = [flip(u_vector(k-nb:k-tau)) flip(y_vector(k-na:k-1))]';

            %Count linearization coefficients.
            [a,b] = networkLinearizationNum(w10, w1, w20, w2, na, nb, tau, x, 'tanh');
            %[a,b] = networkLinearization(w10, w1, w20, w2, na, nb, tau, x, 'tanh');

            %Count step response.
            s = zeros(N,1);
            for j=1:N
               for i=1:min(j,nb)
                   s(j) = s(j) + b(i);
               end
               for i=1:min(j-1,na)
                   s(j) = s(j) - a(i)*s(j-i);
               end
            end

            %Construct dynamic matrix.
            M = zeros(N,Nu);
            for j=1: Nu
               M(:,j) = [zeros(j-1,1); s(1:N-j+1)]';
            end

            K = (M' * M + lambda * eye(Nu))\(M');       
    end
    
    %Count free trajectory.
    switch alg_type
        case 'NO'
            y = w20 + w2*tanh(w10 + w1*[flip(u_vector(k-nb:k-tau)) flip(y_vector(k-na:k-1))]');
            dk = y_vector(1,k) - y;

            y0 = [y_vector(1:k-1) y zeros(1,sim_time+N-k)];
            u0 = [u_vector(1:k-1) zeros(1,sim_time+N-k+1)];
        	traj_fun = @(u) trajectoryFunction(u, y0, u0, k, N, Nu, dk, y_zad, lambda,w20,w2,w10,w1,na,nb,tau);
            
    	case {'NPL', 'GPC'}
            %Construct free trajectory.
            y = w20 + w2*tanh(w10 + w1*[flip(u_vector(k-nb:k-tau)) flip(y_vector(k-na:k-1))]');
            dk = y_vector(1,k) - y;

            y0 = [y_vector(1:k-1) y ones(1,sim_time+N-k)*dk];
            u0 = [u_vector(1:k-1) ones(1,sim_time+Nu-k+1)*u_vector(k-1)];
            for j=1:N
                y0(k+j) = y0(k+j) + w20 + w2*tanh(w10 + w1*[flip(u0(k-nb+j:k-tau+j)) flip(y0(k-na+j:k-1+j))]');
            end

            y0 = y0(k+1:k+N)';
    end
    
    %Count control signal.
    switch alg_type
        case {'NPL', 'GPC'}
            du = K*(y_zad(k)*ones(N,1) - y0);
            u_vector(k) = du(1) + u_vector(k-1);

            if u_vector(k) < u_min
                u_vector(k) = u_min;
            elseif u_vector(k) > u_max
                u_vector(k) = u_max;
            end
           
        case 'PID'
            e_vector(k) = y_zad(k) - y_vector(k);
            u_vector(k) = r2*e_vector(k-2)+r1*e_vector(k-1)+r0*e_vector(k) + u_vector(k-1);
            
            if u_vector(k) < u_min
                u_vector(k) = u_min;
            elseif u_vector(k) > u_max
                u_vector(k) = u_max;
            end
        case 'NO'
            u_start = fmincon(traj_fun,u_start,[],[],[],[],u_min*ones(1,Nu),u_max*ones(1,Nu), [], options);
            u_vector(k) = u_start(1);
    end
end

figure;
% stairs(1:sim_time,u_vector);
hold on;
stairs(1:sim_time,y_vector);
stairs(1:sim_time,y_zad);
hold off;
legend('y', 'y_{zad}');

figure;
stairs(1:sim_time,u_vector);
legend('u');
xlabel('k');
ylabel('y');
