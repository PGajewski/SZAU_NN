clear;

%modelstr = 'OEBFGS';
%modelstr = 'OESD';
modelstr = 'ARXBGFS';

if strcmp('OEBFGS', modelstr)
    K = 1:10;
    models = 1:10;
    [u_val, y_val] = readData('dane_wer.txt');
    [u_learning, y_learning] = readData('dane.txt');
    j = 1;
    ErrorLearning =  ones(size(K,2)*size(models,2), 1);
    ErrorValidate =  ones(size(K,2)*size(models,2), 1);
    Knumber = ones(size(K,2)*size(models,2), 1);
    Modelnumber = ones(size(K,2)*size(models,2), 1);
    tau = 5;
    nb = 6;
    na =2;
    latex = "";
    for k = K
        latex = latex + sprintf('\\begin{center}\r\n    \\begin{tabular}{ |p{1cm}||p{3cm}|p{3cm}||  }\r\n    \\hline\r\n    \\multicolumn{3}{|c|}{K=%d} \\\\\r\n    \\hline\r\n    Próba & B³¹d ucz¹cy &B³¹d weryfikuj¹cy\\\\\r\n    \\hline\r\n    ', k);
        for modeln = models
            Knumber(j) = k;
            Modelnumber(j) = modeln;
            run(sprintf('ModelOEBFGS/K=%d/%d/model.m',k,modeln));
            %run(sprintf('ModelowanieZad2/K=%d/%d/uczenie.m',k,model));
            y_vector = zeros(1,length(u_val));
            y_vector(1:nb) = y_val(1:nb);
            
            for i=(nb)+1:length(u_val)
                y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_val(i-nb:i-tau)) flip(y_vector(i-na:i-1))]');
            end
            ErrorValidate(j,1) =(y_vector-y_val)*(y_vector-y_val)'; %immse(y_vector,y_val);
            
            y_vector = zeros(1,length(u_learning));
            y_vector(1:nb) = y_learning(1:nb);
            for i=(nb)+1:length(u_val)
                y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_learning(i-nb:i-tau)) flip(y_vector(i-na:i-1))]');
            end
            ErrorLearning(j,1) = (y_vector-y_learning)*(y_vector-y_learning)';%immse(y_vector,y_learning);
            latex= latex + sprintf('%d & %.6f & %.6f \\\\\r\n    ', modeln, ErrorLearning(j,1), ErrorValidate(j,1));
            j= j+1;
        end
        latex = latex + sprintf('\\hline\r\n    \\end{tabular}\r\n\\end{center}\r\n');
    end
    ErrorTable = table( Knumber, Modelnumber, ErrorLearning, ErrorValidate);
    best = "";
    for k = K
        minimum = min(ErrorTable.ErrorValidate(Knumber == k));
        best = best + sprintf('%d & %.6f & %.6f \\\\\r\n    ', k, ErrorTable.ErrorLearning(ErrorValidate == minimum), ErrorTable.ErrorValidate(ErrorValidate == minimum));
    end
end
if strcmp('OESD', modelstr)
    K = 4;
    models = 1:10;
    [u_val, y_val] = readData('dane_wer.txt');
    [u_learning, y_learning] = readData('dane.txt');
    j = 1;
    ErrorLearning =  ones(size(K,2)*size(models,2), 1);
    ErrorValidate =  ones(size(K,2)*size(models,2), 1);
    Knumber = ones(size(K,2)*size(models,2), 1);
    Modelnumber = ones(size(K,2)*size(models,2), 1);
    tau = 5;
    nb = 6;
    na =2;
    latex = "";
    for k = K
        latex = latex + sprintf('\\begin{center}\r\n    \\begin{tabular}{ |p{1cm}||p{3cm}|p{3cm}||  }\r\n    \\hline\r\n    \\multicolumn{3}{|c|}{K=%d} \\\\\r\n    \\hline\r\n    Próba & B³¹d ucz¹cy &B³¹d weryfikuj¹cy\\\\\r\n    \\hline\r\n    ', k);
        for modeln = models
            Knumber(j) = k;
            Modelnumber(j) = modeln;
            run(sprintf('ModelOESD/%d/model.m',modeln));
            y_vector = zeros(1,length(u_val));
            y_vector(1:nb) = y_val(1:nb);
            
            for i=(nb)+1:length(u_val)
                y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_val(i-nb:i-tau)) flip(y_vector(i-na:i-1))]');
            end
            ErrorValidate(j,1) =(y_vector-y_val)*(y_vector-y_val)'; %immse(y_vector,y_val);
            
            y_vector = zeros(1,length(u_learning));
            y_vector(1:nb) = y_learning(1:nb);
            for i=(nb)+1:length(u_val)
                y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_learning(i-nb:i-tau)) flip(y_vector(i-na:i-1))]');
            end
            ErrorLearning(j,1) = (y_vector-y_learning)*(y_vector-y_learning)';%immse(y_vector,y_learning);
            latex= latex + sprintf('%d & %.6f & %.6f \\\\\r\n    ', modeln, ErrorLearning(j,1), ErrorValidate(j,1));
            j= j+1;
        end
        latex = latex + sprintf('\\hline\r\n    \\end{tabular}\r\n\\end{center}\r\n');
    end
end


if strcmp('ARXBGFS', modelstr)
    K = 4;
    models = 1:10;
    [u_val, y_val] = readData('dane_wer.txt');
    [u_learning, y_learning] = readData('dane.txt');
    j = 1;
    ErrorLearning =  ones(size(K,2)*size(models,2), 1);
    ErrorValidate =  ones(size(K,2)*size(models,2), 1);
    Knumber = ones(size(K,2)*size(models,2), 1);
    Modelnumber = ones(size(K,2)*size(models,2), 1);
    tau = 5;
    nb = 6;
    na =2;
    latex = "";
    for k = K
        latex = latex + sprintf('\\begin{center}\r\n    \\begin{tabular}{ |p{1cm}||p{3cm}|p{3cm}||  }\r\n    \\hline\r\n    \\multicolumn{3}{|c|}{K=%d} \\\\\r\n    \\hline\r\n    Próba & B³¹d ucz¹cy &B³¹d weryfikuj¹cy\\\\\r\n    \\hline\r\n    ', k);
        for modeln = models
            Knumber(j) = k;
            Modelnumber(j) = modeln;
            run(sprintf('ModelARXBFGS/%d/model.m',modeln));
            y_vector = zeros(1,length(u_val));
            y_vector(1:nb) = y_val(1:nb);
            
            for i=(nb)+1:length(u_val)
                y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_val(i-nb:i-tau)) flip(y_val(i-na:i-1))]');
            end
            ErrorValidate(j,1) =(y_vector-y_val)*(y_vector-y_val)'; %immse(y_vector,y_val);
            
            y_vector = zeros(1,length(u_learning));
            y_vector(1:nb) = y_learning(1:nb);
            for i=(nb)+1:length(u_val)
                 y_vector(1,i) = w20 + w2*tanh(w10 + w1*[flip(u_learning(i-nb:i-tau)) flip(y_learning(i-na:i-1))]');
            end
            ErrorLearning(j,1) = (y_vector-y_learning)*(y_vector-y_learning)';%immse(y_vector,y_learning);
            latex= latex + sprintf('%d & %.6f & %.6f \\\\\r\n    ', modeln, ErrorLearning(j,1), ErrorValidate(j,1));
            j= j+1;
        end
        latex = latex + sprintf('\\hline\r\n    \\end{tabular}\r\n\\end{center}\r\n');
    end
end

