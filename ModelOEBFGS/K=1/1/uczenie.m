%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.789471e+002; foe(n+1)=2.806664e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.781067e+002; foe(n+1)=1.792640e+002; krok(n+1)=4.988661e-004; ng(n+1)=5.045103e+002;
n=2; farx(n+1)=1.776370e+002; foe(n+1)=1.780987e+002; krok(n+1)=3.816075e-003; ng(n+1)=3.154383e+001;
n=3; farx(n+1)=6.086400e+001; foe(n+1)=6.311836e+001; krok(n+1)=5.005537e-002; ng(n+1)=2.337316e+001;
n=4; farx(n+1)=6.049395e+001; foe(n+1)=6.260236e+001; krok(n+1)=3.812452e-004; ng(n+1)=1.657418e+002;
n=5; farx(n+1)=7.309756e+001; foe(n+1)=5.965225e+001; krok(n+1)=2.250273e-001; ng(n+1)=1.789306e+002;
n=6; farx(n+1)=9.126223e+000; foe(n+1)=2.520989e+001; krok(n+1)=1.404256e+000; ng(n+1)=5.785042e+001;
n=7; farx(n+1)=7.373431e+000; foe(n+1)=2.409760e+001; krok(n+1)=6.367965e-001; ng(n+1)=6.481852e+001;
n=8; farx(n+1)=5.416809e+000; foe(n+1)=2.274168e+001; krok(n+1)=3.112246e-001; ng(n+1)=9.643461e+001;
n=9; farx(n+1)=5.238061e+000; foe(n+1)=2.231657e+001; krok(n+1)=6.385257e-002; ng(n+1)=1.169680e+002;
n=10; farx(n+1)=5.612219e+000; foe(n+1)=2.014963e+001; krok(n+1)=4.635137e-001; ng(n+1)=1.310192e+002;
n=11; farx(n+1)=5.785081e+000; foe(n+1)=1.939408e+001; krok(n+1)=4.063898e-001; ng(n+1)=1.383963e+002;
n=12; farx(n+1)=5.604961e+000; foe(n+1)=1.858074e+001; krok(n+1)=2.056099e-001; ng(n+1)=1.316503e+002;
n=13; farx(n+1)=3.733599e+000; foe(n+1)=1.644673e+001; krok(n+1)=1.476581e+000; ng(n+1)=7.432436e+001;
n=14; farx(n+1)=2.271529e+000; foe(n+1)=1.394460e+001; krok(n+1)=2.130938e+000; ng(n+1)=1.417984e+002;
n=15; farx(n+1)=2.215298e+000; foe(n+1)=1.287371e+001; krok(n+1)=2.192961e-001; ng(n+1)=1.387565e+002;
n=16; farx(n+1)=2.108170e+000; foe(n+1)=1.203315e+001; krok(n+1)=4.213257e-001; ng(n+1)=2.244317e+002;
n=17; farx(n+1)=1.787610e+000; foe(n+1)=1.145520e+001; krok(n+1)=4.044487e-001; ng(n+1)=1.570455e+002;
n=18; farx(n+1)=1.266761e+000; foe(n+1)=1.039227e+001; krok(n+1)=8.377756e-001; ng(n+1)=9.697177e+001;
n=19; farx(n+1)=1.212149e+000; foe(n+1)=1.019714e+001; krok(n+1)=4.854776e-001; ng(n+1)=3.030190e+001;
n=20; farx(n+1)=1.308720e+000; foe(n+1)=9.658650e+000; krok(n+1)=1.070120e+000; ng(n+1)=1.051615e+002;
n=21; farx(n+1)=1.249016e+000; foe(n+1)=9.260950e+000; krok(n+1)=9.453922e-001; ng(n+1)=1.129230e+002;
n=22; farx(n+1)=1.208950e+000; foe(n+1)=9.109886e+000; krok(n+1)=1.580303e-001; ng(n+1)=7.167841e+001;
n=23; farx(n+1)=1.121840e+000; foe(n+1)=8.909558e+000; krok(n+1)=5.695627e-001; ng(n+1)=4.911440e+001;
n=24; farx(n+1)=1.016913e+000; foe(n+1)=8.796365e+000; krok(n+1)=4.770341e-001; ng(n+1)=7.265226e+001;
n=25; farx(n+1)=1.040416e+000; foe(n+1)=8.721425e+000; krok(n+1)=1.065517e+000; ng(n+1)=5.013304e+001;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.036788e+000; foe(n+1)=8.714139e+000; krok(n+1)=1.611741e-005; ng(n+1)=4.062298e+001;
n=27; farx(n+1)=1.033405e+000; foe(n+1)=8.706550e+000; krok(n+1)=2.318206e-004; ng(n+1)=1.333133e+001;
n=28; farx(n+1)=1.043623e+000; foe(n+1)=8.695044e+000; krok(n+1)=5.769389e-004; ng(n+1)=1.040408e+001;
n=29; farx(n+1)=1.008214e+000; foe(n+1)=8.655582e+000; krok(n+1)=8.442897e-003; ng(n+1)=5.589365e+000;
n=30; farx(n+1)=1.049715e+000; foe(n+1)=8.594986e+000; krok(n+1)=4.256083e-002; ng(n+1)=1.002131e+001;
n=31; farx(n+1)=9.403544e-001; foe(n+1)=8.556831e+000; krok(n+1)=4.992693e-001; ng(n+1)=4.051923e+001;
n=32; farx(n+1)=9.196658e-001; foe(n+1)=8.529643e+000; krok(n+1)=8.131859e-001; ng(n+1)=4.949075e+001;
n=33; farx(n+1)=8.741187e-001; foe(n+1)=8.499602e+000; krok(n+1)=1.498210e+000; ng(n+1)=2.247809e+001;
n=34; farx(n+1)=8.675890e-001; foe(n+1)=8.488320e+000; krok(n+1)=3.739988e-001; ng(n+1)=2.528418e+001;
n=35; farx(n+1)=8.438062e-001; foe(n+1)=8.475570e+000; krok(n+1)=1.202875e+000; ng(n+1)=2.553318e+001;
n=36; farx(n+1)=7.485509e-001; foe(n+1)=8.459164e+000; krok(n+1)=1.836100e+000; ng(n+1)=2.771443e+001;
n=37; farx(n+1)=7.204432e-001; foe(n+1)=8.451826e+000; krok(n+1)=8.043035e-001; ng(n+1)=9.167727e+000;
n=38; farx(n+1)=6.990886e-001; foe(n+1)=8.449409e+000; krok(n+1)=1.333102e+000; ng(n+1)=8.503026e+000;
n=39; farx(n+1)=6.947075e-001; foe(n+1)=8.448093e+000; krok(n+1)=5.900599e-001; ng(n+1)=2.090526e+001;
n=40; farx(n+1)=6.689340e-001; foe(n+1)=8.444927e+000; krok(n+1)=4.726961e-001; ng(n+1)=1.758450e+001;
n=41; farx(n+1)=6.623762e-001; foe(n+1)=8.443310e+000; krok(n+1)=1.119298e+000; ng(n+1)=8.136212e+000;
n=42; farx(n+1)=6.708928e-001; foe(n+1)=8.442751e+000; krok(n+1)=7.560816e-001; ng(n+1)=5.795035e+000;
n=43; farx(n+1)=6.524949e-001; foe(n+1)=8.442191e+000; krok(n+1)=1.885812e+000; ng(n+1)=3.288872e+000;
n=44; farx(n+1)=6.354997e-001; foe(n+1)=8.441669e+000; krok(n+1)=9.270274e-001; ng(n+1)=6.515010e+000;
n=45; farx(n+1)=6.225650e-001; foe(n+1)=8.441190e+000; krok(n+1)=1.352447e+000; ng(n+1)=7.534063e+000;
n=46; farx(n+1)=6.237516e-001; foe(n+1)=8.441047e+000; krok(n+1)=9.639418e-001; ng(n+1)=1.668471e+000;
n=47; farx(n+1)=6.181065e-001; foe(n+1)=8.440977e+000; krok(n+1)=6.698571e-001; ng(n+1)=1.210118e+000;
n=48; farx(n+1)=6.165238e-001; foe(n+1)=8.440967e+000; krok(n+1)=9.072770e-001; ng(n+1)=1.593517e+000;
n=49; farx(n+1)=6.160641e-001; foe(n+1)=8.440966e+000; krok(n+1)=7.655551e-001; ng(n+1)=6.495892e-002;
n=50; farx(n+1)=6.162007e-001; foe(n+1)=8.440966e+000; krok(n+1)=1.422745e+000; ng(n+1)=1.547159e-001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)