%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.822400e+002; foe(n+1)=2.819634e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.790116e+002; foe(n+1)=1.784618e+002; krok(n+1)=5.014459e-004; ng(n+1)=4.813714e+002;
n=2; farx(n+1)=6.973601e+001; foe(n+1)=6.435159e+001; krok(n+1)=1.178633e-001; ng(n+1)=1.280062e+001;
n=3; farx(n+1)=6.960699e+001; foe(n+1)=6.422240e+001; krok(n+1)=4.554340e-005; ng(n+1)=1.346051e+002;
n=4; farx(n+1)=8.060982e+001; foe(n+1)=6.182944e+001; krok(n+1)=1.752988e-001; ng(n+1)=1.441497e+002;
n=5; farx(n+1)=6.839188e+001; foe(n+1)=5.992828e+001; krok(n+1)=4.884576e-001; ng(n+1)=7.648101e+001;
n=6; farx(n+1)=6.212086e+001; foe(n+1)=5.698416e+001; krok(n+1)=3.105330e-002; ng(n+1)=1.170708e+002;
n=7; farx(n+1)=2.153726e+001; foe(n+1)=4.437318e+001; krok(n+1)=9.872312e-001; ng(n+1)=1.014257e+002;
n=8; farx(n+1)=1.611872e+001; foe(n+1)=4.128404e+001; krok(n+1)=4.954706e-002; ng(n+1)=3.714804e+002;
n=9; farx(n+1)=1.509008e+001; foe(n+1)=3.664939e+001; krok(n+1)=7.437068e-002; ng(n+1)=4.610566e+002;
n=10; farx(n+1)=1.471133e+001; foe(n+1)=2.961977e+001; krok(n+1)=1.794664e-001; ng(n+1)=2.920510e+002;
n=11; farx(n+1)=1.221529e+001; foe(n+1)=2.745711e+001; krok(n+1)=5.312625e-001; ng(n+1)=1.165557e+002;
n=12; farx(n+1)=7.223601e+000; foe(n+1)=2.407252e+001; krok(n+1)=1.476581e+000; ng(n+1)=1.420426e+002;
n=13; farx(n+1)=3.798937e+000; foe(n+1)=2.250646e+001; krok(n+1)=9.072770e-001; ng(n+1)=1.172721e+002;
n=14; farx(n+1)=2.399875e+000; foe(n+1)=1.800280e+001; krok(n+1)=9.205345e-001; ng(n+1)=1.843753e+002;
n=15; farx(n+1)=2.840441e+000; foe(n+1)=1.662458e+001; krok(n+1)=8.172294e-002; ng(n+1)=4.169616e+002;
n=16; farx(n+1)=3.066765e+000; foe(n+1)=1.543044e+001; krok(n+1)=1.694093e-001; ng(n+1)=3.606147e+002;
n=17; farx(n+1)=2.301237e+000; foe(n+1)=1.252892e+001; krok(n+1)=1.447720e+000; ng(n+1)=3.007924e+002;
n=18; farx(n+1)=2.227647e+000; foe(n+1)=1.231399e+001; krok(n+1)=4.228884e-001; ng(n+1)=1.370596e+002;
n=19; farx(n+1)=1.619656e+000; foe(n+1)=1.061892e+001; krok(n+1)=2.704893e+000; ng(n+1)=4.278117e+001;
n=20; farx(n+1)=1.282769e+000; foe(n+1)=1.007339e+001; krok(n+1)=1.307567e+000; ng(n+1)=1.015719e+002;
n=21; farx(n+1)=1.179892e+000; foe(n+1)=9.612974e+000; krok(n+1)=1.154449e+000; ng(n+1)=4.213766e+001;
n=22; farx(n+1)=1.189748e+000; foe(n+1)=9.371610e+000; krok(n+1)=2.073757e-001; ng(n+1)=3.459386e+001;
n=23; farx(n+1)=1.175458e+000; foe(n+1)=9.198846e+000; krok(n+1)=6.019322e-001; ng(n+1)=6.559458e+001;
n=24; farx(n+1)=1.218686e+000; foe(n+1)=9.060684e+000; krok(n+1)=3.691454e-001; ng(n+1)=1.105855e+002;
n=25; farx(n+1)=1.162891e+000; foe(n+1)=8.921527e+000; krok(n+1)=7.413928e-001; ng(n+1)=5.719074e+001;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.158559e+000; foe(n+1)=8.914625e+000; krok(n+1)=2.352777e-005; ng(n+1)=3.438547e+001;
n=27; farx(n+1)=1.128555e+000; foe(n+1)=8.881275e+000; krok(n+1)=5.469456e-004; ng(n+1)=1.701795e+001;
n=28; farx(n+1)=1.123372e+000; foe(n+1)=8.862732e+000; krok(n+1)=2.594055e-004; ng(n+1)=1.850544e+001;
n=29; farx(n+1)=1.103641e+000; foe(n+1)=8.803399e+000; krok(n+1)=2.398987e-002; ng(n+1)=3.145849e+000;
n=30; farx(n+1)=1.153623e+000; foe(n+1)=8.736656e+000; krok(n+1)=1.763378e-002; ng(n+1)=2.869051e+001;
n=31; farx(n+1)=1.004833e+000; foe(n+1)=8.655143e+000; krok(n+1)=4.836593e-001; ng(n+1)=3.448700e+001;
n=32; farx(n+1)=9.632612e-001; foe(n+1)=8.602165e+000; krok(n+1)=7.386536e-001; ng(n+1)=5.590687e+001;
n=33; farx(n+1)=9.197745e-001; foe(n+1)=8.528523e+000; krok(n+1)=1.885812e+000; ng(n+1)=2.850164e+001;
n=34; farx(n+1)=9.175407e-001; foe(n+1)=8.511107e+000; krok(n+1)=2.991387e-001; ng(n+1)=3.941835e+001;
n=35; farx(n+1)=8.721565e-001; foe(n+1)=8.489794e+000; krok(n+1)=1.062525e+000; ng(n+1)=2.775544e+001;
n=36; farx(n+1)=8.030667e-001; foe(n+1)=8.475159e+000; krok(n+1)=8.773192e-001; ng(n+1)=3.749605e+001;
n=37; farx(n+1)=7.708062e-001; foe(n+1)=8.462014e+000; krok(n+1)=8.289013e-001; ng(n+1)=1.287626e+001;
n=38; farx(n+1)=7.527688e-001; foe(n+1)=8.457064e+000; krok(n+1)=4.536903e-001; ng(n+1)=1.728636e+001;
n=39; farx(n+1)=7.340098e-001; foe(n+1)=8.452789e+000; krok(n+1)=6.290754e-001; ng(n+1)=1.317711e+001;
n=40; farx(n+1)=6.885187e-001; foe(n+1)=8.449476e+000; krok(n+1)=1.422745e+000; ng(n+1)=1.838287e+001;
n=41; farx(n+1)=6.854467e-001; foe(n+1)=8.447050e+000; krok(n+1)=1.210739e+000; ng(n+1)=1.018768e+001;
n=42; farx(n+1)=6.689362e-001; foe(n+1)=8.444479e+000; krok(n+1)=7.178655e-001; ng(n+1)=9.171692e+000;
n=43; farx(n+1)=6.647017e-001; foe(n+1)=8.443494e+000; krok(n+1)=6.273059e-001; ng(n+1)=5.721491e+000;
n=44; farx(n+1)=6.553635e-001; foe(n+1)=8.442477e+000; krok(n+1)=7.461831e-001; ng(n+1)=8.529739e+000;
n=45; farx(n+1)=6.579436e-001; foe(n+1)=8.442132e+000; krok(n+1)=6.329112e-001; ng(n+1)=5.965776e+000;
n=46; farx(n+1)=6.456668e-001; foe(n+1)=8.441725e+000; krok(n+1)=1.683587e+000; ng(n+1)=3.841076e+000;
n=47; farx(n+1)=6.267347e-001; foe(n+1)=8.441191e+000; krok(n+1)=2.315580e+000; ng(n+1)=8.021064e+000;
n=48; farx(n+1)=6.191345e-001; foe(n+1)=8.441038e+000; krok(n+1)=1.031099e+000; ng(n+1)=2.439106e+000;
n=49; farx(n+1)=6.152394e-001; foe(n+1)=8.441005e+000; krok(n+1)=1.001154e+000; ng(n+1)=1.941822e+000;
n=50; farx(n+1)=6.154396e-001; foe(n+1)=8.440979e+000; krok(n+1)=7.525649e-001; ng(n+1)=2.312669e+000;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
