%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.104743e+002; foe(n+1)=2.161316e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.765369e+002; foe(n+1)=1.810296e+002; krok(n+1)=5.277180e-004; ng(n+1)=7.689844e+002;
n=2; farx(n+1)=7.721754e+001; foe(n+1)=9.025954e+001; krok(n+1)=1.254679e-002; ng(n+1)=4.247585e+002;
n=3; farx(n+1)=4.562765e+001; foe(n+1)=5.884941e+001; krok(n+1)=3.927263e-004; ng(n+1)=1.563374e+003;
n=4; farx(n+1)=2.457063e+001; foe(n+1)=5.390210e+001; krok(n+1)=9.416683e-003; ng(n+1)=1.126286e+002;
n=5; farx(n+1)=2.332629e+001; foe(n+1)=5.354764e+001; krok(n+1)=3.831728e-004; ng(n+1)=9.900046e+002;
n=6; farx(n+1)=9.267056e+000; foe(n+1)=3.246983e+001; krok(n+1)=9.326355e-003; ng(n+1)=1.203986e+003;
n=7; farx(n+1)=9.366508e+000; foe(n+1)=2.909776e+001; krok(n+1)=1.112427e-003; ng(n+1)=1.203621e+003;
n=8; farx(n+1)=1.021188e+001; foe(n+1)=2.509860e+001; krok(n+1)=1.162226e-003; ng(n+1)=1.597437e+003;
n=9; farx(n+1)=8.869622e+000; foe(n+1)=2.156369e+001; krok(n+1)=2.399905e-002; ng(n+1)=8.751848e+002;
n=10; farx(n+1)=8.736002e+000; foe(n+1)=2.111072e+001; krok(n+1)=1.135356e-003; ng(n+1)=6.296027e+002;
n=11; farx(n+1)=8.827324e+000; foe(n+1)=2.047135e+001; krok(n+1)=2.817506e-003; ng(n+1)=4.749134e+002;
n=12; farx(n+1)=8.799006e+000; foe(n+1)=2.001213e+001; krok(n+1)=7.088910e-003; ng(n+1)=3.831380e+002;
n=13; farx(n+1)=6.450144e+000; foe(n+1)=1.553911e+001; krok(n+1)=7.173481e-002; ng(n+1)=3.885036e+002;
n=14; farx(n+1)=5.298146e+000; foe(n+1)=1.356427e+001; krok(n+1)=7.799544e-003; ng(n+1)=2.600138e+002;
n=15; farx(n+1)=5.086444e+000; foe(n+1)=1.286806e+001; krok(n+1)=2.716429e-004; ng(n+1)=8.343861e+002;
n=16; farx(n+1)=4.642883e+000; foe(n+1)=1.185907e+001; krok(n+1)=1.464637e-003; ng(n+1)=5.951975e+002;
n=17; farx(n+1)=4.194736e+000; foe(n+1)=1.018525e+001; krok(n+1)=4.812762e-003; ng(n+1)=5.126023e+002;
n=18; farx(n+1)=3.601615e+000; foe(n+1)=9.029170e+000; krok(n+1)=1.438322e-002; ng(n+1)=6.486576e+002;
n=19; farx(n+1)=3.149987e+000; foe(n+1)=8.455228e+000; krok(n+1)=2.129617e-003; ng(n+1)=6.418989e+002;
n=20; farx(n+1)=2.803227e+000; foe(n+1)=7.882562e+000; krok(n+1)=3.799123e-003; ng(n+1)=9.770796e+002;
n=21; farx(n+1)=2.422331e+000; foe(n+1)=7.189945e+000; krok(n+1)=6.669234e-003; ng(n+1)=4.265926e+002;
n=22; farx(n+1)=1.780347e+000; foe(n+1)=6.169556e+000; krok(n+1)=3.300845e-002; ng(n+1)=2.107482e+002;
n=23; farx(n+1)=1.530131e+000; foe(n+1)=5.810467e+000; krok(n+1)=2.831403e-003; ng(n+1)=7.887583e+002;
n=24; farx(n+1)=1.268108e+000; foe(n+1)=5.286598e+000; krok(n+1)=5.767896e-003; ng(n+1)=2.781647e+002;
n=25; farx(n+1)=1.095124e+000; foe(n+1)=4.838281e+000; krok(n+1)=1.274938e-002; ng(n+1)=2.854972e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.075455e+000; foe(n+1)=4.444913e+000; krok(n+1)=4.588605e-006; ng(n+1)=1.059045e+003;
n=27; farx(n+1)=1.084176e+000; foe(n+1)=4.414135e+000; krok(n+1)=2.769106e-005; ng(n+1)=1.305725e+002;
n=28; farx(n+1)=1.101313e+000; foe(n+1)=4.229142e+000; krok(n+1)=4.616173e-004; ng(n+1)=8.566472e+001;
n=29; farx(n+1)=1.083268e+000; foe(n+1)=4.083128e+000; krok(n+1)=3.952630e-004; ng(n+1)=9.916930e+001;
n=30; farx(n+1)=1.014002e+000; foe(n+1)=3.133862e+000; krok(n+1)=4.027729e-003; ng(n+1)=7.489353e+001;
n=31; farx(n+1)=9.793836e-001; foe(n+1)=2.548451e+000; krok(n+1)=1.042882e-002; ng(n+1)=8.496413e+002;
n=32; farx(n+1)=9.533113e-001; foe(n+1)=2.153686e+000; krok(n+1)=1.586357e-002; ng(n+1)=2.626318e+002;
n=33; farx(n+1)=9.068208e-001; foe(n+1)=1.952017e+000; krok(n+1)=6.728198e-003; ng(n+1)=2.992185e+002;
n=34; farx(n+1)=8.372622e-001; foe(n+1)=1.836595e+000; krok(n+1)=6.884742e-003; ng(n+1)=2.574780e+002;
n=35; farx(n+1)=7.839379e-001; foe(n+1)=1.706525e+000; krok(n+1)=6.800237e-003; ng(n+1)=1.966877e+002;
n=36; farx(n+1)=7.111505e-001; foe(n+1)=1.561314e+000; krok(n+1)=2.606636e-002; ng(n+1)=1.802673e+002;
n=37; farx(n+1)=6.575824e-001; foe(n+1)=1.466075e+000; krok(n+1)=1.363555e-002; ng(n+1)=1.774471e+002;
n=38; farx(n+1)=6.223244e-001; foe(n+1)=1.400133e+000; krok(n+1)=9.127324e-003; ng(n+1)=1.761587e+002;
n=39; farx(n+1)=5.740711e-001; foe(n+1)=1.308421e+000; krok(n+1)=1.525498e-002; ng(n+1)=1.828535e+002;
n=40; farx(n+1)=5.440094e-001; foe(n+1)=1.251708e+000; krok(n+1)=1.640492e-002; ng(n+1)=1.561731e+002;
n=41; farx(n+1)=5.012419e-001; foe(n+1)=1.137200e+000; krok(n+1)=4.633705e-002; ng(n+1)=1.509810e+002;
n=42; farx(n+1)=4.905532e-001; foe(n+1)=1.058050e+000; krok(n+1)=2.539936e-002; ng(n+1)=8.515024e+001;
n=43; farx(n+1)=4.782219e-001; foe(n+1)=1.013040e+000; krok(n+1)=1.547817e-002; ng(n+1)=1.666489e+002;
n=44; farx(n+1)=4.782806e-001; foe(n+1)=9.394356e-001; krok(n+1)=6.045742e-002; ng(n+1)=1.212728e+002;
n=45; farx(n+1)=4.807249e-001; foe(n+1)=9.165162e-001; krok(n+1)=2.693892e-002; ng(n+1)=1.164182e+002;
n=46; farx(n+1)=5.189385e-001; foe(n+1)=8.861566e-001; krok(n+1)=8.877087e-002; ng(n+1)=7.449964e+001;
n=47; farx(n+1)=5.297457e-001; foe(n+1)=8.642743e-001; krok(n+1)=5.514490e-002; ng(n+1)=1.223420e+002;
n=48; farx(n+1)=5.299702e-001; foe(n+1)=8.567125e-001; krok(n+1)=9.447410e-003; ng(n+1)=6.287793e+001;
n=49; farx(n+1)=5.213126e-001; foe(n+1)=8.306983e-001; krok(n+1)=5.524306e-002; ng(n+1)=9.356521e+001;
n=50; farx(n+1)=4.926851e-001; foe(n+1)=7.880269e-001; krok(n+1)=2.203031e-001; ng(n+1)=7.632876e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
