%uczenie predyktora arx
clear all;
n=0; farx(n+1)=2.227229e+002; foe(n+1)=2.243677e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.746708e+002; foe(n+1)=1.772905e+002; krok(n+1)=5.113081e-004; ng(n+1)=5.708252e+002;
n=2; farx(n+1)=3.589078e+001; foe(n+1)=7.307133e+001; krok(n+1)=1.589439e-002; ng(n+1)=2.386635e+002;
n=3; farx(n+1)=2.570626e+001; foe(n+1)=5.910379e+001; krok(n+1)=1.164626e-003; ng(n+1)=4.290377e+002;
n=4; farx(n+1)=1.889748e+001; foe(n+1)=4.986305e+001; krok(n+1)=1.694223e-003; ng(n+1)=3.132154e+002;
n=5; farx(n+1)=3.866918e+000; foe(n+1)=1.326448e+002; krok(n+1)=4.792105e-003; ng(n+1)=3.046693e+002;
n=6; farx(n+1)=1.735974e+000; foe(n+1)=2.130086e+001; krok(n+1)=2.113019e-002; ng(n+1)=2.707454e+002;
n=7; farx(n+1)=1.048467e+000; foe(n+1)=5.840746e+001; krok(n+1)=3.280985e-002; ng(n+1)=1.534411e+002;
n=8; farx(n+1)=5.782459e-001; foe(n+1)=1.547404e+001; krok(n+1)=4.155078e-002; ng(n+1)=9.460631e+001;
n=9; farx(n+1)=5.065472e-001; foe(n+1)=1.150812e+001; krok(n+1)=3.172714e-002; ng(n+1)=5.177527e+001;
n=10; farx(n+1)=4.454481e-001; foe(n+1)=1.128850e+001; krok(n+1)=2.230542e-001; ng(n+1)=9.880132e+000;
n=11; farx(n+1)=4.385313e-001; foe(n+1)=1.019492e+001; krok(n+1)=2.165649e-002; ng(n+1)=1.199775e+001;
n=12; farx(n+1)=4.128580e-001; foe(n+1)=1.261100e+001; krok(n+1)=3.768236e-001; ng(n+1)=1.176967e+001;
n=13; farx(n+1)=3.766147e-001; foe(n+1)=1.250672e+001; krok(n+1)=3.180725e-001; ng(n+1)=5.358614e+000;
n=14; farx(n+1)=3.259788e-001; foe(n+1)=7.891529e+000; krok(n+1)=1.434696e-001; ng(n+1)=2.646300e+001;
n=15; farx(n+1)=3.037400e-001; foe(n+1)=1.034176e+001; krok(n+1)=2.097829e-001; ng(n+1)=9.190538e+000;
n=16; farx(n+1)=2.816158e-001; foe(n+1)=9.108246e+000; krok(n+1)=9.763190e-001; ng(n+1)=5.261102e+000;
n=17; farx(n+1)=2.759611e-001; foe(n+1)=8.546650e+000; krok(n+1)=2.905057e-001; ng(n+1)=1.237628e+001;
n=18; farx(n+1)=2.715771e-001; foe(n+1)=8.931295e+000; krok(n+1)=5.002089e-001; ng(n+1)=5.802896e+000;
n=19; farx(n+1)=2.640107e-001; foe(n+1)=7.711881e+000; krok(n+1)=2.392207e+000; ng(n+1)=3.526445e+000;
n=20; farx(n+1)=2.548570e-001; foe(n+1)=4.924927e+000; krok(n+1)=2.765168e+000; ng(n+1)=2.782748e+000;
n=21; farx(n+1)=2.505822e-001; foe(n+1)=5.179706e+000; krok(n+1)=2.893371e-001; ng(n+1)=5.739524e+000;
n=22; farx(n+1)=2.470614e-001; foe(n+1)=4.118922e+000; krok(n+1)=5.520515e-001; ng(n+1)=5.805271e+000;
n=23; farx(n+1)=2.460615e-001; foe(n+1)=3.887884e+000; krok(n+1)=3.255844e-001; ng(n+1)=3.117839e+000;
n=24; farx(n+1)=2.453702e-001; foe(n+1)=3.951833e+000; krok(n+1)=3.268918e-001; ng(n+1)=3.170630e+000;
n=25; farx(n+1)=2.441055e-001; foe(n+1)=3.458637e+000; krok(n+1)=1.754638e+000; ng(n+1)=1.082965e+000;
%odnowa zmiennej metryki
n=26; farx(n+1)=2.437684e-001; foe(n+1)=3.494574e+000; krok(n+1)=5.391128e-004; ng(n+1)=3.045938e+000;
n=27; farx(n+1)=2.437361e-001; foe(n+1)=3.484508e+000; krok(n+1)=1.667309e-003; ng(n+1)=5.767081e-001;
n=28; farx(n+1)=2.436296e-001; foe(n+1)=3.582883e+000; krok(n+1)=1.137613e-003; ng(n+1)=1.006148e+000;
n=29; farx(n+1)=2.430771e-001; foe(n+1)=3.421538e+000; krok(n+1)=3.026447e-001; ng(n+1)=1.556238e-001;
n=30; farx(n+1)=2.420054e-001; foe(n+1)=3.881041e+000; krok(n+1)=2.538306e-001; ng(n+1)=3.969449e-001;
n=31; farx(n+1)=2.414567e-001; foe(n+1)=3.627406e+000; krok(n+1)=8.097460e-002; ng(n+1)=3.615059e+000;
n=32; farx(n+1)=2.383363e-001; foe(n+1)=2.967856e+000; krok(n+1)=7.491049e-001; ng(n+1)=5.874078e+000;
n=33; farx(n+1)=2.358518e-001; foe(n+1)=2.680693e+000; krok(n+1)=5.045908e-001; ng(n+1)=2.982662e+000;
n=34; farx(n+1)=2.334460e-001; foe(n+1)=2.607462e+000; krok(n+1)=6.228822e-001; ng(n+1)=2.849368e+000;
n=35; farx(n+1)=2.293580e-001; foe(n+1)=3.078710e+000; krok(n+1)=6.449478e-001; ng(n+1)=4.158564e+000;
n=36; farx(n+1)=2.288051e-001; foe(n+1)=3.287274e+000; krok(n+1)=6.504518e-002; ng(n+1)=3.001833e+000;
n=37; farx(n+1)=2.267758e-001; foe(n+1)=2.765710e+000; krok(n+1)=1.249330e+000; ng(n+1)=4.222601e-001;
n=38; farx(n+1)=2.247733e-001; foe(n+1)=2.313805e+000; krok(n+1)=3.403631e-001; ng(n+1)=4.128843e+000;
n=39; farx(n+1)=2.238024e-001; foe(n+1)=2.136969e+000; krok(n+1)=1.934788e-001; ng(n+1)=5.869956e+000;
n=40; farx(n+1)=2.216739e-001; foe(n+1)=2.358292e+000; krok(n+1)=7.248392e-001; ng(n+1)=8.085860e-001;
n=41; farx(n+1)=2.197115e-001; foe(n+1)=2.413298e+000; krok(n+1)=5.657309e-001; ng(n+1)=3.617069e+000;
n=42; farx(n+1)=2.166939e-001; foe(n+1)=2.104871e+000; krok(n+1)=4.188592e-001; ng(n+1)=7.595389e+000;
n=43; farx(n+1)=2.149993e-001; foe(n+1)=1.868793e+000; krok(n+1)=1.565625e-001; ng(n+1)=3.675944e+000;
n=44; farx(n+1)=2.128621e-001; foe(n+1)=1.793759e+000; krok(n+1)=3.691454e-001; ng(n+1)=1.625134e+000;
n=45; farx(n+1)=2.114607e-001; foe(n+1)=1.864095e+000; krok(n+1)=2.186286e-001; ng(n+1)=5.690522e+000;
n=46; farx(n+1)=2.099538e-001; foe(n+1)=1.439307e+000; krok(n+1)=2.209722e-001; ng(n+1)=4.512053e+000;
n=47; farx(n+1)=2.073651e-001; foe(n+1)=1.254406e+000; krok(n+1)=3.337224e-001; ng(n+1)=4.108032e+000;
n=48; farx(n+1)=2.060912e-001; foe(n+1)=1.177358e+000; krok(n+1)=7.847380e-001; ng(n+1)=3.072123e+000;
n=49; farx(n+1)=2.047722e-001; foe(n+1)=1.033666e+000; krok(n+1)=1.056270e+000; ng(n+1)=1.611857e+000;
n=50; farx(n+1)=2.042553e-001; foe(n+1)=9.983194e-001; krok(n+1)=3.838379e-001; ng(n+1)=2.682634e+000;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora ARX');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
