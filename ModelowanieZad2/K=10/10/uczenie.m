%uczenie predyktora oe
clear all;
n=0; farx(n+1)=3.214663e+002; foe(n+1)=3.092617e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.706756e+002; foe(n+1)=1.594926e+002; krok(n+1)=5.080539e-004; ng(n+1)=2.010632e+003;
n=2; farx(n+1)=6.919134e+001; foe(n+1)=5.836749e+001; krok(n+1)=4.829711e-003; ng(n+1)=6.973255e+002;
n=3; farx(n+1)=6.610686e+001; foe(n+1)=5.791791e+001; krok(n+1)=3.485603e-003; ng(n+1)=7.858491e+001;
n=4; farx(n+1)=3.279242e+001; foe(n+1)=4.867372e+001; krok(n+1)=2.502768e-002; ng(n+1)=1.070232e+002;
n=5; farx(n+1)=1.960539e+001; foe(n+1)=4.299966e+001; krok(n+1)=1.779752e-003; ng(n+1)=4.307817e+002;
n=6; farx(n+1)=1.138614e+001; foe(n+1)=3.519288e+001; krok(n+1)=3.550402e-003; ng(n+1)=8.641306e+002;
n=7; farx(n+1)=9.964764e+000; foe(n+1)=3.201887e+001; krok(n+1)=2.881993e-004; ng(n+1)=1.685888e+003;
n=8; farx(n+1)=1.070526e+001; foe(n+1)=2.830555e+001; krok(n+1)=3.846158e-003; ng(n+1)=2.203159e+003;
n=9; farx(n+1)=1.104812e+001; foe(n+1)=2.689531e+001; krok(n+1)=8.374265e-004; ng(n+1)=2.100408e+003;
n=10; farx(n+1)=1.158951e+001; foe(n+1)=2.485762e+001; krok(n+1)=1.080144e-002; ng(n+1)=1.441004e+003;
n=11; farx(n+1)=1.085222e+001; foe(n+1)=2.116641e+001; krok(n+1)=1.810600e-003; ng(n+1)=1.108164e+003;
n=12; farx(n+1)=1.035854e+001; foe(n+1)=2.061682e+001; krok(n+1)=3.516052e-003; ng(n+1)=1.507058e+002;
n=13; farx(n+1)=8.210259e+000; foe(n+1)=1.734870e+001; krok(n+1)=2.570124e-002; ng(n+1)=2.127370e+002;
n=14; farx(n+1)=5.827859e+000; foe(n+1)=1.369630e+001; krok(n+1)=8.323975e-003; ng(n+1)=6.272594e+002;
n=15; farx(n+1)=5.487468e+000; foe(n+1)=1.291559e+001; krok(n+1)=3.868579e-003; ng(n+1)=2.750357e+002;
n=16; farx(n+1)=4.168885e+000; foe(n+1)=1.073379e+001; krok(n+1)=7.444102e-003; ng(n+1)=5.874328e+002;
n=17; farx(n+1)=3.713086e+000; foe(n+1)=9.527606e+000; krok(n+1)=7.528887e-004; ng(n+1)=1.402103e+003;
n=18; farx(n+1)=3.164419e+000; foe(n+1)=7.536812e+000; krok(n+1)=1.706649e-002; ng(n+1)=1.169833e+003;
n=19; farx(n+1)=2.925019e+000; foe(n+1)=6.833719e+000; krok(n+1)=6.970662e-003; ng(n+1)=2.536927e+002;
n=20; farx(n+1)=2.538503e+000; foe(n+1)=6.320103e+000; krok(n+1)=2.778799e-003; ng(n+1)=4.106405e+002;
n=21; farx(n+1)=1.473086e+000; foe(n+1)=4.909092e+000; krok(n+1)=6.099924e-003; ng(n+1)=3.967548e+002;
n=22; farx(n+1)=1.221606e+000; foe(n+1)=4.441409e+000; krok(n+1)=1.875750e-003; ng(n+1)=8.064309e+002;
n=23; farx(n+1)=9.726272e-001; foe(n+1)=4.048155e+000; krok(n+1)=1.136199e-002; ng(n+1)=3.839773e+002;
n=24; farx(n+1)=8.684046e-001; foe(n+1)=3.725964e+000; krok(n+1)=8.669283e-003; ng(n+1)=2.890239e+002;
n=25; farx(n+1)=7.859519e-001; foe(n+1)=3.481903e+000; krok(n+1)=8.443488e-003; ng(n+1)=5.176896e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=7.924770e-001; foe(n+1)=3.466580e+000; krok(n+1)=8.683747e-005; ng(n+1)=4.888760e+001;
n=27; farx(n+1)=7.890373e-001; foe(n+1)=3.454840e+000; krok(n+1)=7.131570e-005; ng(n+1)=4.683669e+001;
n=28; farx(n+1)=7.884163e-001; foe(n+1)=3.433021e+000; krok(n+1)=4.645269e-005; ng(n+1)=1.218333e+002;
n=29; farx(n+1)=7.895350e-001; foe(n+1)=3.097322e+000; krok(n+1)=1.899828e-003; ng(n+1)=6.737927e+001;
n=30; farx(n+1)=7.648349e-001; foe(n+1)=2.210915e+000; krok(n+1)=1.797739e-002; ng(n+1)=7.405669e+001;
n=31; farx(n+1)=7.288349e-001; foe(n+1)=2.094059e+000; krok(n+1)=5.167850e-003; ng(n+1)=1.039306e+002;
n=32; farx(n+1)=6.116078e-001; foe(n+1)=1.645584e+000; krok(n+1)=6.669234e-003; ng(n+1)=2.784801e+002;
n=33; farx(n+1)=5.866552e-001; foe(n+1)=1.532313e+000; krok(n+1)=1.121665e-002; ng(n+1)=3.088573e+002;
n=34; farx(n+1)=5.793709e-001; foe(n+1)=1.431861e+000; krok(n+1)=1.021537e-002; ng(n+1)=1.179339e+002;
n=35; farx(n+1)=5.679645e-001; foe(n+1)=1.301525e+000; krok(n+1)=7.883344e-003; ng(n+1)=1.772940e+002;
n=36; farx(n+1)=5.603376e-001; foe(n+1)=1.250141e+000; krok(n+1)=9.082851e-003; ng(n+1)=9.879630e+001;
n=37; farx(n+1)=5.109462e-001; foe(n+1)=1.117752e+000; krok(n+1)=8.321100e-002; ng(n+1)=3.748828e+001;
n=38; farx(n+1)=4.845781e-001; foe(n+1)=9.853781e-001; krok(n+1)=5.999763e-003; ng(n+1)=2.822081e+002;
n=39; farx(n+1)=4.740458e-001; foe(n+1)=9.407068e-001; krok(n+1)=7.760194e-003; ng(n+1)=1.218766e+002;
n=40; farx(n+1)=4.387997e-001; foe(n+1)=8.315337e-001; krok(n+1)=2.896961e-002; ng(n+1)=1.980030e+002;
n=41; farx(n+1)=4.335948e-001; foe(n+1)=8.150233e-001; krok(n+1)=5.999763e-003; ng(n+1)=1.146934e+002;
n=42; farx(n+1)=4.000253e-001; foe(n+1)=7.650732e-001; krok(n+1)=4.095813e-002; ng(n+1)=8.622131e+001;
n=43; farx(n+1)=3.811816e-001; foe(n+1)=6.996427e-001; krok(n+1)=4.614317e-002; ng(n+1)=1.177501e+002;
n=44; farx(n+1)=3.777873e-001; foe(n+1)=6.685439e-001; krok(n+1)=1.017451e-002; ng(n+1)=1.350527e+002;
n=45; farx(n+1)=3.688312e-001; foe(n+1)=6.462687e-001; krok(n+1)=3.631321e-002; ng(n+1)=5.333030e+001;
n=46; farx(n+1)=3.657623e-001; foe(n+1)=6.325614e-001; krok(n+1)=7.626318e-003; ng(n+1)=1.104913e+002;
n=47; farx(n+1)=3.766841e-001; foe(n+1)=6.062707e-001; krok(n+1)=1.079418e-001; ng(n+1)=5.766894e+001;
n=48; farx(n+1)=3.689820e-001; foe(n+1)=5.897915e-001; krok(n+1)=6.213826e-002; ng(n+1)=1.260471e+002;
n=49; farx(n+1)=3.702429e-001; foe(n+1)=5.762648e-001; krok(n+1)=6.128758e-002; ng(n+1)=9.643336e+001;
n=50; farx(n+1)=3.762159e-001; foe(n+1)=5.622728e-001; krok(n+1)=7.864394e-002; ng(n+1)=5.527602e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
