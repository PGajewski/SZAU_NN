%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.037472e+002; foe(n+1)=2.064154e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.781150e+002; foe(n+1)=1.808221e+002; krok(n+1)=5.591620e-004; ng(n+1)=6.826675e+002;
n=2; farx(n+1)=1.038941e+002; foe(n+1)=1.062113e+002; krok(n+1)=1.097075e-002; ng(n+1)=3.914786e+002;
n=3; farx(n+1)=5.479089e+001; foe(n+1)=6.021268e+001; krok(n+1)=6.089906e-004; ng(n+1)=1.264166e+003;
n=4; farx(n+1)=4.550299e+001; foe(n+1)=5.899872e+001; krok(n+1)=4.696827e-003; ng(n+1)=8.496536e+001;
n=5; farx(n+1)=3.620340e+001; foe(n+1)=5.759331e+001; krok(n+1)=9.064776e-004; ng(n+1)=3.403194e+002;
n=6; farx(n+1)=2.132831e+001; foe(n+1)=4.162391e+001; krok(n+1)=5.560984e-002; ng(n+1)=6.254200e+002;
n=7; farx(n+1)=1.931705e+001; foe(n+1)=3.928568e+001; krok(n+1)=1.460933e-003; ng(n+1)=5.859462e+002;
n=8; farx(n+1)=1.625153e+001; foe(n+1)=3.460226e+001; krok(n+1)=3.568597e-003; ng(n+1)=6.876613e+002;
n=9; farx(n+1)=1.239754e+001; foe(n+1)=3.013942e+001; krok(n+1)=8.510569e-003; ng(n+1)=4.029643e+002;
n=10; farx(n+1)=1.057653e+001; foe(n+1)=2.723499e+001; krok(n+1)=5.381741e-003; ng(n+1)=6.273075e+002;
n=11; farx(n+1)=9.229112e+000; foe(n+1)=2.339102e+001; krok(n+1)=2.008899e-002; ng(n+1)=6.574418e+002;
n=12; farx(n+1)=9.000464e+000; foe(n+1)=2.220736e+001; krok(n+1)=3.334617e-003; ng(n+1)=3.865278e+002;
n=13; farx(n+1)=8.742324e+000; foe(n+1)=2.170806e+001; krok(n+1)=5.608389e-003; ng(n+1)=1.579325e+002;
n=14; farx(n+1)=7.790131e+000; foe(n+1)=2.068787e+001; krok(n+1)=6.227174e-003; ng(n+1)=1.750754e+002;
n=15; farx(n+1)=6.461398e+000; foe(n+1)=1.788522e+001; krok(n+1)=7.737158e-003; ng(n+1)=3.783851e+002;
n=16; farx(n+1)=6.235792e+000; foe(n+1)=1.744046e+001; krok(n+1)=2.825558e-004; ng(n+1)=7.516397e+002;
n=17; farx(n+1)=6.031834e+000; foe(n+1)=1.695203e+001; krok(n+1)=1.282071e-003; ng(n+1)=4.177202e+002;
n=18; farx(n+1)=5.011094e+000; foe(n+1)=1.618770e+001; krok(n+1)=7.799544e-003; ng(n+1)=1.208993e+002;
n=19; farx(n+1)=4.763791e+000; foe(n+1)=1.581892e+001; krok(n+1)=8.948804e-003; ng(n+1)=4.202403e+002;
n=20; farx(n+1)=3.524438e+000; foe(n+1)=1.260812e+001; krok(n+1)=4.076532e-002; ng(n+1)=6.141376e+002;
n=21; farx(n+1)=2.947829e+000; foe(n+1)=1.069412e+001; krok(n+1)=2.129131e-003; ng(n+1)=6.620988e+002;
n=22; farx(n+1)=2.561982e+000; foe(n+1)=8.233337e+000; krok(n+1)=1.369522e-003; ng(n+1)=1.444490e+003;
n=23; farx(n+1)=2.574315e+000; foe(n+1)=7.122920e+000; krok(n+1)=1.250908e-003; ng(n+1)=1.784493e+003;
n=24; farx(n+1)=2.631763e+000; foe(n+1)=6.516439e+000; krok(n+1)=4.336062e-003; ng(n+1)=1.245333e+003;
n=25; farx(n+1)=2.352838e+000; foe(n+1)=5.563830e+000; krok(n+1)=9.150227e-003; ng(n+1)=1.001965e+003;
%odnowa zmiennej metryki
n=26; farx(n+1)=2.334418e+000; foe(n+1)=5.234241e+000; krok(n+1)=1.348707e-005; ng(n+1)=5.557173e+002;
n=27; farx(n+1)=2.364603e+000; foe(n+1)=5.086450e+000; krok(n+1)=2.981309e-005; ng(n+1)=2.726228e+002;
n=28; farx(n+1)=2.334726e+000; foe(n+1)=4.779714e+000; krok(n+1)=6.831655e-005; ng(n+1)=2.311959e+002;
n=29; farx(n+1)=1.434733e+000; foe(n+1)=2.129055e+000; krok(n+1)=4.629191e-003; ng(n+1)=1.048477e+002;
n=30; farx(n+1)=1.146789e+000; foe(n+1)=1.743173e+000; krok(n+1)=5.296999e-003; ng(n+1)=4.322913e+002;
n=31; farx(n+1)=9.710593e-001; foe(n+1)=1.609414e+000; krok(n+1)=2.778799e-003; ng(n+1)=1.900509e+002;
n=32; farx(n+1)=8.419331e-001; foe(n+1)=1.516990e+000; krok(n+1)=1.779752e-003; ng(n+1)=2.120011e+002;
n=33; farx(n+1)=7.071173e-001; foe(n+1)=1.407333e+000; krok(n+1)=6.606617e-003; ng(n+1)=1.283898e+002;
n=34; farx(n+1)=6.821413e-001; foe(n+1)=1.358026e+000; krok(n+1)=2.440235e-002; ng(n+1)=1.293945e+002;
n=35; farx(n+1)=6.657111e-001; foe(n+1)=1.278236e+000; krok(n+1)=6.533169e-002; ng(n+1)=2.343452e+001;
n=36; farx(n+1)=6.278803e-001; foe(n+1)=1.155791e+000; krok(n+1)=1.438335e-002; ng(n+1)=1.947111e+002;
n=37; farx(n+1)=6.038910e-001; foe(n+1)=1.128887e+000; krok(n+1)=9.385969e-003; ng(n+1)=8.933834e+001;
n=38; farx(n+1)=5.651597e-001; foe(n+1)=1.097446e+000; krok(n+1)=4.285764e-002; ng(n+1)=1.526294e+002;
n=39; farx(n+1)=5.710678e-001; foe(n+1)=1.048221e+000; krok(n+1)=1.079418e-001; ng(n+1)=7.655020e+001;
n=40; farx(n+1)=5.575606e-001; foe(n+1)=1.030683e+000; krok(n+1)=1.450364e-002; ng(n+1)=8.212412e+001;
n=41; farx(n+1)=5.438531e-001; foe(n+1)=1.006548e+000; krok(n+1)=3.475615e-003; ng(n+1)=1.214698e+002;
n=42; farx(n+1)=4.902376e-001; foe(n+1)=9.200934e-001; krok(n+1)=2.387143e-002; ng(n+1)=1.179733e+002;
n=43; farx(n+1)=4.805653e-001; foe(n+1)=8.975440e-001; krok(n+1)=2.946581e-002; ng(n+1)=9.469741e+001;
n=44; farx(n+1)=4.745112e-001; foe(n+1)=8.776270e-001; krok(n+1)=8.571529e-002; ng(n+1)=5.795677e+001;
n=45; farx(n+1)=4.478248e-001; foe(n+1)=8.516461e-001; krok(n+1)=7.549232e-002; ng(n+1)=7.942313e+001;
n=46; farx(n+1)=4.329131e-001; foe(n+1)=8.356469e-001; krok(n+1)=6.863735e-002; ng(n+1)=3.815969e+001;
n=47; farx(n+1)=3.989594e-001; foe(n+1)=8.044421e-001; krok(n+1)=1.255673e-001; ng(n+1)=5.856138e+001;
n=48; farx(n+1)=3.685626e-001; foe(n+1)=7.701604e-001; krok(n+1)=6.528949e-002; ng(n+1)=1.171346e+002;
n=49; farx(n+1)=3.682622e-001; foe(n+1)=7.339462e-001; krok(n+1)=1.677703e-001; ng(n+1)=6.578157e+001;
n=50; farx(n+1)=3.755157e-001; foe(n+1)=7.134053e-001; krok(n+1)=2.779047e-002; ng(n+1)=1.467067e+002;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
