%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.126809e+002; foe(n+1)=2.117833e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.753503e+002; foe(n+1)=1.747853e+002; krok(n+1)=5.469456e-004; ng(n+1)=6.998289e+002;
n=2; farx(n+1)=6.628116e+001; foe(n+1)=6.189863e+001; krok(n+1)=1.058808e-002; ng(n+1)=4.048221e+002;
n=3; farx(n+1)=6.098289e+001; foe(n+1)=6.137826e+001; krok(n+1)=2.827499e-003; ng(n+1)=7.081751e+001;
n=4; farx(n+1)=3.469910e+000; foe(n+1)=2.944597e+001; krok(n+1)=1.053841e-002; ng(n+1)=1.672404e+002;
n=5; farx(n+1)=3.191267e+000; foe(n+1)=2.919945e+001; krok(n+1)=8.058704e-006; ng(n+1)=3.566961e+003;
n=6; farx(n+1)=3.140791e+000; foe(n+1)=2.915364e+001; krok(n+1)=1.697983e-004; ng(n+1)=4.170416e+003;
n=7; farx(n+1)=3.715657e+000; foe(n+1)=2.650223e+001; krok(n+1)=1.337344e-002; ng(n+1)=4.150098e+003;
n=8; farx(n+1)=4.682619e+000; foe(n+1)=2.174712e+001; krok(n+1)=1.183257e-003; ng(n+1)=3.439979e+003;
n=9; farx(n+1)=5.604156e+000; foe(n+1)=1.934895e+001; krok(n+1)=7.043764e-004; ng(n+1)=1.921847e+003;
n=10; farx(n+1)=6.136877e+000; foe(n+1)=1.747731e+001; krok(n+1)=9.618292e-003; ng(n+1)=1.338504e+003;
n=11; farx(n+1)=5.950440e+000; foe(n+1)=1.620791e+001; krok(n+1)=1.019133e-002; ng(n+1)=4.940362e+002;
n=12; farx(n+1)=5.828588e+000; foe(n+1)=1.471613e+001; krok(n+1)=1.038090e-002; ng(n+1)=5.499836e+002;
n=13; farx(n+1)=5.509041e+000; foe(n+1)=1.408778e+001; krok(n+1)=9.045235e-003; ng(n+1)=2.470525e+002;
n=14; farx(n+1)=5.037581e+000; foe(n+1)=1.343207e+001; krok(n+1)=1.394089e-002; ng(n+1)=3.328109e+002;
n=15; farx(n+1)=3.866772e+000; foe(n+1)=1.206301e+001; krok(n+1)=8.116945e-003; ng(n+1)=2.581124e+002;
n=16; farx(n+1)=3.436285e+000; foe(n+1)=1.152438e+001; krok(n+1)=1.730147e-003; ng(n+1)=6.439703e+002;
n=17; farx(n+1)=2.931160e+000; foe(n+1)=1.072914e+001; krok(n+1)=1.708036e-003; ng(n+1)=7.901152e+002;
n=18; farx(n+1)=2.383357e+000; foe(n+1)=9.511189e+000; krok(n+1)=7.954572e-003; ng(n+1)=3.583439e+002;
n=19; farx(n+1)=2.084934e+000; foe(n+1)=8.842651e+000; krok(n+1)=6.854056e-003; ng(n+1)=7.253402e+002;
n=20; farx(n+1)=1.777772e+000; foe(n+1)=7.885401e+000; krok(n+1)=2.187783e-003; ng(n+1)=1.205199e+003;
n=21; farx(n+1)=1.548869e+000; foe(n+1)=7.089054e+000; krok(n+1)=1.095618e-002; ng(n+1)=4.069333e+002;
n=22; farx(n+1)=1.345304e+000; foe(n+1)=6.388943e+000; krok(n+1)=6.970444e-003; ng(n+1)=3.989931e+002;
n=23; farx(n+1)=1.123067e+000; foe(n+1)=5.727710e+000; krok(n+1)=5.186798e-003; ng(n+1)=5.637462e+002;
n=24; farx(n+1)=7.717374e-001; foe(n+1)=4.393302e+000; krok(n+1)=1.417782e-002; ng(n+1)=1.257277e+003;
n=25; farx(n+1)=6.508939e-001; foe(n+1)=3.252239e+000; krok(n+1)=1.142218e-002; ng(n+1)=1.485910e+003;
%odnowa zmiennej metryki
n=26; farx(n+1)=6.491223e-001; foe(n+1)=3.248132e+000; krok(n+1)=3.999817e-005; ng(n+1)=4.986178e+001;
n=27; farx(n+1)=6.460319e-001; foe(n+1)=3.238914e+000; krok(n+1)=2.519594e-005; ng(n+1)=8.635817e+001;
n=28; farx(n+1)=6.520246e-001; foe(n+1)=3.190000e+000; krok(n+1)=2.215284e-004; ng(n+1)=7.647431e+001;
n=29; farx(n+1)=5.726900e-001; foe(n+1)=2.545067e+000; krok(n+1)=2.531421e-003; ng(n+1)=7.665153e+001;
n=30; farx(n+1)=4.971896e-001; foe(n+1)=2.097722e+000; krok(n+1)=6.042575e-003; ng(n+1)=4.488104e+002;
n=31; farx(n+1)=4.954065e-001; foe(n+1)=2.004150e+000; krok(n+1)=1.631651e-003; ng(n+1)=4.584237e+002;
n=32; farx(n+1)=4.872312e-001; foe(n+1)=1.731460e+000; krok(n+1)=1.532189e-002; ng(n+1)=1.849059e+002;
n=33; farx(n+1)=5.419104e-001; foe(n+1)=1.570537e+000; krok(n+1)=1.527445e-002; ng(n+1)=1.301565e+002;
n=34; farx(n+1)=5.641856e-001; foe(n+1)=1.502612e+000; krok(n+1)=5.139053e-003; ng(n+1)=2.553585e+002;
n=35; farx(n+1)=5.399164e-001; foe(n+1)=1.396627e+000; krok(n+1)=2.038894e-002; ng(n+1)=1.033316e+002;
n=36; farx(n+1)=4.854958e-001; foe(n+1)=1.261741e+000; krok(n+1)=3.280985e-002; ng(n+1)=1.473482e+002;
n=37; farx(n+1)=4.840500e-001; foe(n+1)=1.231511e+000; krok(n+1)=7.242402e-003; ng(n+1)=2.168812e+002;
n=38; farx(n+1)=4.515573e-001; foe(n+1)=1.194995e+000; krok(n+1)=2.559636e-002; ng(n+1)=3.450783e+001;
n=39; farx(n+1)=4.326034e-001; foe(n+1)=1.108128e+000; krok(n+1)=2.008899e-002; ng(n+1)=1.812445e+002;
n=40; farx(n+1)=4.264289e-001; foe(n+1)=1.088807e+000; krok(n+1)=7.191676e-003; ng(n+1)=9.820686e+001;
n=41; farx(n+1)=4.248402e-001; foe(n+1)=1.040328e+000; krok(n+1)=4.086147e-002; ng(n+1)=2.322982e+002;
n=42; farx(n+1)=4.216899e-001; foe(n+1)=9.966651e-001; krok(n+1)=9.393654e-003; ng(n+1)=1.479346e+002;
n=43; farx(n+1)=4.287085e-001; foe(n+1)=9.570015e-001; krok(n+1)=3.034189e-002; ng(n+1)=1.687456e+002;
n=44; farx(n+1)=4.437556e-001; foe(n+1)=9.181898e-001; krok(n+1)=6.602029e-002; ng(n+1)=8.313410e+001;
n=45; farx(n+1)=4.542328e-001; foe(n+1)=8.942061e-001; krok(n+1)=6.539800e-002; ng(n+1)=4.910046e+001;
n=46; farx(n+1)=4.489412e-001; foe(n+1)=8.697465e-001; krok(n+1)=7.280720e-002; ng(n+1)=1.314422e+002;
n=47; farx(n+1)=4.301798e-001; foe(n+1)=8.548890e-001; krok(n+1)=6.995615e-002; ng(n+1)=3.102629e+001;
n=48; farx(n+1)=4.021961e-001; foe(n+1)=8.321831e-001; krok(n+1)=9.809225e-002; ng(n+1)=6.114999e+001;
n=49; farx(n+1)=3.787288e-001; foe(n+1)=7.884552e-001; krok(n+1)=2.363480e-001; ng(n+1)=1.333176e+002;
n=50; farx(n+1)=3.726525e-001; foe(n+1)=7.770238e-001; krok(n+1)=3.522595e-002; ng(n+1)=7.538806e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
