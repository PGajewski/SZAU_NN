%%Data
clear all;
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
for i =1:length(u)
   y_stat(i) = StaticNonlinearObject(u(i)); 
end

figure(1);
plot(u,y_stat);
title('Ch-ka statyczna obiektu');
xlabel('u');
ylabel('y_{stat}');

%% Generate data for neural network.
seeds = [16, 50];
files = {'dane.txt','dane_wer.txt'};
plot_texts = {'Dane uczace','Dane weryfikujace'};
file_template = '%f %f\n';

%k=1 - learning data
%k=2 - verifying data
for k=1:2
    rng(seeds(k));
    rng
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
            y_vector(1,j) = g2(x_vector(1,j));
        end
    end

    %Display data.
    figure(k+1);
    plot(1:length(y_vector),y_vector);
    hold on;
    plot(1:length(u_vector),u_vector);
    hold off;
    title(plot_texts{k});
    legend('y','u');
    xlabel('t');
    ylabel('y,u');
    
    %Save to file
    fileID = fopen(files{k},'w');
    for i=1:length(u_vector)
        fprintf(fileID,file_template,u_vector(i),y_vector(i));
    end
    fclose(fileID);
end

%% Validate object.
% Read config.
[tau, nb, na, K, max_iter, error, alghorithm] = readConfig();

% Read data.
[u_val, y_val] = readData('dane_wer.txt');
[u_learning, y_learning] = readData('dane.txt');

save('tmp.mat');
%Load model of network.
model;

%Display learning data.
uczenie
load('tmp.mat');
delete('tmp.mat');

%Count response of network for validate data.
y_vector = zeros(1,length(u_val));
x_vector = zeros(2,length(u_val));

%Init state
x_vector(:,1) = [0;0];
    
for i=(u_delay)+1:length(u_val)
        x_vector(1,i+1) = -alfa1 * x_vector(1,i) + x_vector(2,i) + beta1*g1(u_val(1,i-u_delay));
        x_vector(2,i+1) = -alfa2 * x_vector(1,i) + beta2*g1(u_val(1,i-u_delay));
        y_vector(1,i) = g2(x_vector(1,i));
end

%Compare model for validating data.
figure(5);
hold on;
plot(1:length(y_vector),y_vector);
plot(1:length(y_val),y_val);
plot(1:length(u_val),u_val);
hold off;
title('Compare for validating data');
legend('y_{val}','y_{model}','u');
xlabel('t');
ylabel('y,u');

%Count MSE.
err = immse(y_vector,y_val)