%uczenie predyktora oe
clear all;
n=0; farx(n+1)=1.902423e+002; foe(n+1)=1.943539e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.598506e+002; foe(n+1)=1.637780e+002; krok(n+1)=5.418302e-004; ng(n+1)=7.509603e+002;
n=2; farx(n+1)=6.166727e+001; foe(n+1)=6.192437e+001; krok(n+1)=5.829338e-003; ng(n+1)=4.563882e+002;
n=3; farx(n+1)=5.728130e+001; foe(n+1)=5.576694e+001; krok(n+1)=2.105249e-003; ng(n+1)=3.508365e+002;
n=4; farx(n+1)=5.123696e+001; foe(n+1)=5.393372e+001; krok(n+1)=1.284535e-002; ng(n+1)=7.422961e+001;
n=5; farx(n+1)=1.356265e+001; foe(n+1)=3.117971e+001; krok(n+1)=1.844475e-002; ng(n+1)=2.269565e+002;
n=6; farx(n+1)=1.091496e+001; foe(n+1)=2.453805e+001; krok(n+1)=3.636899e-004; ng(n+1)=1.165050e+003;
n=7; farx(n+1)=1.010834e+001; foe(n+1)=2.336973e+001; krok(n+1)=1.191677e-003; ng(n+1)=8.961862e+002;
n=8; farx(n+1)=9.308767e+000; foe(n+1)=2.080867e+001; krok(n+1)=5.405350e-004; ng(n+1)=8.720255e+002;
n=9; farx(n+1)=7.573750e+000; foe(n+1)=1.932682e+001; krok(n+1)=9.744063e-003; ng(n+1)=1.536109e+002;
n=10; farx(n+1)=6.526698e+000; foe(n+1)=1.673778e+001; krok(n+1)=1.495979e-002; ng(n+1)=5.230715e+002;
n=11; farx(n+1)=6.324802e+000; foe(n+1)=1.594400e+001; krok(n+1)=2.354171e-003; ng(n+1)=3.843966e+002;
n=12; farx(n+1)=5.418314e+000; foe(n+1)=1.401822e+001; krok(n+1)=5.711090e-003; ng(n+1)=3.310820e+002;
n=13; farx(n+1)=4.500497e+000; foe(n+1)=1.264134e+001; krok(n+1)=4.633998e-003; ng(n+1)=5.280112e+002;
n=14; farx(n+1)=3.139578e+000; foe(n+1)=7.186451e+000; krok(n+1)=9.899545e-003; ng(n+1)=6.914375e+002;
n=15; farx(n+1)=3.005577e+000; foe(n+1)=6.934583e+000; krok(n+1)=7.368272e-004; ng(n+1)=3.195560e+002;
n=16; farx(n+1)=2.694326e+000; foe(n+1)=5.845394e+000; krok(n+1)=5.478088e-003; ng(n+1)=2.586460e+002;
n=17; farx(n+1)=2.264127e+000; foe(n+1)=5.175449e+000; krok(n+1)=6.019169e-003; ng(n+1)=2.907802e+002;
n=18; farx(n+1)=2.207481e+000; foe(n+1)=5.077627e+000; krok(n+1)=1.597856e-003; ng(n+1)=2.279604e+002;
n=19; farx(n+1)=1.749529e+000; foe(n+1)=4.329310e+000; krok(n+1)=2.008899e-002; ng(n+1)=1.825302e+002;
n=20; farx(n+1)=1.410478e+000; foe(n+1)=3.690025e+000; krok(n+1)=1.660195e-002; ng(n+1)=4.585676e+002;
n=21; farx(n+1)=1.239529e+000; foe(n+1)=3.138291e+000; krok(n+1)=2.378798e-002; ng(n+1)=9.690346e+001;
n=22; farx(n+1)=1.096436e+000; foe(n+1)=2.817735e+000; krok(n+1)=9.428457e-003; ng(n+1)=1.634162e+002;
n=23; farx(n+1)=9.803938e-001; foe(n+1)=2.638341e+000; krok(n+1)=9.545881e-003; ng(n+1)=1.569416e+002;
n=24; farx(n+1)=8.213360e-001; foe(n+1)=2.325327e+000; krok(n+1)=7.366454e-003; ng(n+1)=3.368790e+002;
n=25; farx(n+1)=8.021308e-001; foe(n+1)=2.238582e+000; krok(n+1)=1.017451e-002; ng(n+1)=1.540916e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=7.967046e-001; foe(n+1)=2.205475e+000; krok(n+1)=1.355752e-005; ng(n+1)=2.041302e+002;
n=27; farx(n+1)=8.015001e-001; foe(n+1)=2.191045e+000; krok(n+1)=4.506169e-005; ng(n+1)=6.963211e+001;
n=28; farx(n+1)=7.710317e-001; foe(n+1)=2.074538e+000; krok(n+1)=1.881412e-003; ng(n+1)=3.544810e+001;
n=29; farx(n+1)=7.531819e-001; foe(n+1)=2.054661e+000; krok(n+1)=2.044712e-004; ng(n+1)=4.544195e+001;
n=30; farx(n+1)=6.550727e-001; foe(n+1)=1.751856e+000; krok(n+1)=2.865057e-002; ng(n+1)=2.305771e+001;
n=31; farx(n+1)=6.458985e-001; foe(n+1)=1.668700e+000; krok(n+1)=6.130766e-003; ng(n+1)=1.599509e+002;
n=32; farx(n+1)=6.505287e-001; foe(n+1)=1.597036e+000; krok(n+1)=1.153579e-002; ng(n+1)=9.444725e+001;
n=33; farx(n+1)=6.533667e-001; foe(n+1)=1.388359e+000; krok(n+1)=2.710756e-002; ng(n+1)=2.524758e+002;
n=34; farx(n+1)=6.163087e-001; foe(n+1)=1.259297e+000; krok(n+1)=3.153338e-002; ng(n+1)=6.010141e+001;
n=35; farx(n+1)=6.056094e-001; foe(n+1)=1.230630e+000; krok(n+1)=4.587536e-003; ng(n+1)=1.315923e+002;
n=36; farx(n+1)=5.066628e-001; foe(n+1)=1.091062e+000; krok(n+1)=2.788178e-002; ng(n+1)=4.222015e+001;
n=37; farx(n+1)=4.607197e-001; foe(n+1)=1.005710e+000; krok(n+1)=4.766707e-003; ng(n+1)=2.425573e+002;
n=38; farx(n+1)=4.454357e-001; foe(n+1)=9.616186e-001; krok(n+1)=1.254246e-002; ng(n+1)=5.851864e+001;
n=39; farx(n+1)=4.378452e-001; foe(n+1)=9.327316e-001; krok(n+1)=7.632150e-003; ng(n+1)=1.275692e+002;
n=40; farx(n+1)=4.206127e-001; foe(n+1)=8.862038e-001; krok(n+1)=2.421456e-002; ng(n+1)=5.462778e+001;
n=41; farx(n+1)=4.045994e-001; foe(n+1)=8.393259e-001; krok(n+1)=4.285764e-002; ng(n+1)=1.106943e+002;
n=42; farx(n+1)=3.954947e-001; foe(n+1)=7.664546e-001; krok(n+1)=5.955282e-002; ng(n+1)=1.100278e+002;
n=43; farx(n+1)=3.920177e-001; foe(n+1)=7.313806e-001; krok(n+1)=4.732206e-002; ng(n+1)=4.821680e+001;
n=44; farx(n+1)=3.886755e-001; foe(n+1)=6.889074e-001; krok(n+1)=3.897540e-002; ng(n+1)=1.209283e+002;
n=45; farx(n+1)=3.867468e-001; foe(n+1)=6.614718e-001; krok(n+1)=5.625683e-002; ng(n+1)=1.623783e+002;
n=46; farx(n+1)=4.056933e-001; foe(n+1)=6.204190e-001; krok(n+1)=4.523999e-002; ng(n+1)=7.207035e+001;
n=47; farx(n+1)=3.988055e-001; foe(n+1)=6.086491e-001; krok(n+1)=6.090562e-002; ng(n+1)=7.586828e+001;
n=48; farx(n+1)=3.802854e-001; foe(n+1)=5.836806e-001; krok(n+1)=1.261335e-001; ng(n+1)=5.811431e+001;
n=49; farx(n+1)=3.606017e-001; foe(n+1)=5.589473e-001; krok(n+1)=6.053393e-002; ng(n+1)=1.056987e+002;
n=50; farx(n+1)=3.516718e-001; foe(n+1)=5.253664e-001; krok(n+1)=8.452076e-002; ng(n+1)=7.373458e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
