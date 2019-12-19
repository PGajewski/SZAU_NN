clear;
K = 1:10;
models = 1:10;
[u_val, y_val] = readData('dane_wer.txt');
[u_learning, y_learning] = readData('dane.txt');
j = 1;
ErrorLearning =  ones(100, 1);
ErrorValidate =  ones(100, 1);
Knumber = ones(100, 1);
Modelnumber = ones(100, 1);
tau = 5;
nb = 7;
na =2;
latex = "";
for k = K
    latex = latex + sprintf('\\begin{center}\r\n    \\begin{tabular}{ |p{1cm}||p{3cm}|p{3cm}||  }\r\n    \\hline\r\n    \\multicolumn{3}{|c|}{K=%d} \\\\\r\n    \\hline\r\n    Pr�ba & B��d ucz�cy &B��d weryfikuj�cy\\\\\r\n    \\hline\r\n    ', k);
    for modeln = models
        Knumber(j) = k;
        Modelnumber(j) = modeln;
        run(sprintf('ModelowanieZad2/K=%d/%d/model.m',k,modeln));
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
for K = 1:10
    minimum = min(ErrorTable.ErrorValidate(Knumber == K));
    best = best + sprintf('%d & %.6f & %.6f \\\\\r\n    ', K, ErrorTable.ErrorLearning(ErrorValidate == minimum), ErrorTable.ErrorValidate(ErrorValidate == minimum));
end

