%uczenie predyktora oe
clear all;
n=0; farx(n+1)=3.099075e+002; foe(n+1)=3.038755e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.789959e+002; foe(n+1)=1.762932e+002; krok(n+1)=5.018139e-004; ng(n+1)=7.473284e+002;
n=2; farx(n+1)=6.534433e+001; foe(n+1)=6.632156e+001; krok(n+1)=2.501951e-002; ng(n+1)=8.834069e+001;
n=3; farx(n+1)=6.278157e+001; foe(n+1)=6.331005e+001; krok(n+1)=7.010487e-004; ng(n+1)=2.287115e+002;
n=4; farx(n+1)=6.162127e+001; foe(n+1)=6.040285e+001; krok(n+1)=3.036370e-003; ng(n+1)=2.243554e+002;
n=5; farx(n+1)=2.592814e+001; foe(n+1)=5.240968e+001; krok(n+1)=3.220929e-002; ng(n+1)=4.521511e+001;
n=6; farx(n+1)=1.155954e+001; foe(n+1)=4.829097e+001; krok(n+1)=2.487263e-003; ng(n+1)=5.151048e+002;
n=7; farx(n+1)=6.353181e+000; foe(n+1)=4.637918e+001; krok(n+1)=1.317301e-003; ng(n+1)=1.353966e+003;
n=8; farx(n+1)=4.468093e+000; foe(n+1)=4.426410e+001; krok(n+1)=1.037622e-003; ng(n+1)=2.625636e+003;
n=9; farx(n+1)=4.425013e+000; foe(n+1)=4.392172e+001; krok(n+1)=3.758726e-003; ng(n+1)=4.688438e+003;
n=10; farx(n+1)=4.495418e+000; foe(n+1)=4.358445e+001; krok(n+1)=1.667309e-003; ng(n+1)=5.172874e+003;
n=11; farx(n+1)=4.857314e+000; foe(n+1)=4.197317e+001; krok(n+1)=7.145002e-003; ng(n+1)=5.110972e+003;
n=12; farx(n+1)=4.931988e+000; foe(n+1)=4.176820e+001; krok(n+1)=4.927090e-004; ng(n+1)=3.151346e+003;
n=13; farx(n+1)=5.545488e+000; foe(n+1)=4.085633e+001; krok(n+1)=4.372259e-003; ng(n+1)=2.571190e+003;
n=14; farx(n+1)=5.478094e+000; foe(n+1)=3.694485e+001; krok(n+1)=7.841683e-002; ng(n+1)=2.397779e+003;
n=15; farx(n+1)=3.678247e+000; foe(n+1)=2.885856e+001; krok(n+1)=2.720095e-002; ng(n+1)=3.682117e+003;
n=16; farx(n+1)=3.584716e+000; foe(n+1)=1.928853e+001; krok(n+1)=3.300845e-002; ng(n+1)=3.110977e+003;
n=17; farx(n+1)=2.838361e+000; foe(n+1)=1.799663e+001; krok(n+1)=4.879939e-002; ng(n+1)=4.025997e+002;
n=18; farx(n+1)=2.054880e+000; foe(n+1)=1.221421e+001; krok(n+1)=5.458698e-001; ng(n+1)=4.026797e+002;
n=19; farx(n+1)=1.539583e+000; foe(n+1)=1.040855e+001; krok(n+1)=8.610892e-001; ng(n+1)=2.564854e+002;
n=20; farx(n+1)=1.714559e+000; foe(n+1)=9.763015e+000; krok(n+1)=8.858351e-001; ng(n+1)=2.174801e+002;
n=21; farx(n+1)=1.598318e+000; foe(n+1)=8.790586e+000; krok(n+1)=7.215816e-001; ng(n+1)=7.805336e+001;
n=22; farx(n+1)=1.583594e+000; foe(n+1)=7.955301e+000; krok(n+1)=4.686024e-001; ng(n+1)=3.687009e+002;
n=23; farx(n+1)=1.359779e+000; foe(n+1)=7.640925e+000; krok(n+1)=4.536903e-001; ng(n+1)=9.879912e+001;
n=24; farx(n+1)=1.079324e+000; foe(n+1)=7.204381e+000; krok(n+1)=8.658973e-001; ng(n+1)=1.603713e+002;
n=25; farx(n+1)=8.484442e-001; foe(n+1)=6.464158e+000; krok(n+1)=4.923082e-001; ng(n+1)=1.534868e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=8.457504e-001; foe(n+1)=6.321484e+000; krok(n+1)=6.560802e-006; ng(n+1)=3.061059e+002;
n=27; farx(n+1)=8.451968e-001; foe(n+1)=6.260388e+000; krok(n+1)=4.999290e-005; ng(n+1)=1.135411e+002;
n=28; farx(n+1)=8.865686e-001; foe(n+1)=6.184857e+000; krok(n+1)=2.224361e-004; ng(n+1)=5.280779e+001;
n=29; farx(n+1)=8.444665e-001; foe(n+1)=5.304017e+000; krok(n+1)=7.702766e-003; ng(n+1)=3.514675e+001;
n=30; farx(n+1)=8.257718e-001; foe(n+1)=4.518104e+000; krok(n+1)=4.546520e-003; ng(n+1)=3.792570e+002;
n=31; farx(n+1)=8.212671e-001; foe(n+1)=4.224374e+000; krok(n+1)=9.725768e-003; ng(n+1)=2.473806e+002;
n=32; farx(n+1)=7.889197e-001; foe(n+1)=3.775663e+000; krok(n+1)=1.056509e-002; ng(n+1)=1.084798e+002;
n=33; farx(n+1)=8.036174e-001; foe(n+1)=3.648968e+000; krok(n+1)=1.734832e-002; ng(n+1)=2.107238e+002;
n=34; farx(n+1)=9.081303e-001; foe(n+1)=3.499330e+000; krok(n+1)=9.050552e-003; ng(n+1)=1.696835e+002;
n=35; farx(n+1)=9.063173e-001; foe(n+1)=3.399076e+000; krok(n+1)=1.778431e-001; ng(n+1)=1.204401e+002;
n=36; farx(n+1)=7.590444e-001; foe(n+1)=3.067101e+000; krok(n+1)=2.656312e-001; ng(n+1)=1.972167e+002;
n=37; farx(n+1)=7.579306e-001; foe(n+1)=2.901874e+000; krok(n+1)=1.620110e-001; ng(n+1)=1.900068e+002;
n=38; farx(n+1)=7.159057e-001; foe(n+1)=2.788631e+000; krok(n+1)=2.317569e-001; ng(n+1)=1.056958e+002;
n=39; farx(n+1)=6.457895e-001; foe(n+1)=2.611599e+000; krok(n+1)=4.500546e-001; ng(n+1)=1.743492e+002;
n=40; farx(n+1)=7.164487e-001; foe(n+1)=2.536616e+000; krok(n+1)=5.219391e-001; ng(n+1)=4.571665e+001;
n=41; farx(n+1)=6.472844e-001; foe(n+1)=2.506112e+000; krok(n+1)=3.329123e-001; ng(n+1)=7.943365e+001;
n=42; farx(n+1)=6.330096e-001; foe(n+1)=2.494320e+000; krok(n+1)=3.867248e-001; ng(n+1)=3.821774e+001;
n=43; farx(n+1)=6.251753e-001; foe(n+1)=2.489282e+000; krok(n+1)=8.008859e-001; ng(n+1)=2.898457e+001;
n=44; farx(n+1)=6.388510e-001; foe(n+1)=2.484021e+000; krok(n+1)=1.707324e+000; ng(n+1)=6.167754e+000;
n=45; farx(n+1)=6.501944e-001; foe(n+1)=2.481675e+000; krok(n+1)=7.847380e-001; ng(n+1)=1.972943e+001;
n=46; farx(n+1)=6.506471e-001; foe(n+1)=2.478078e+000; krok(n+1)=8.224396e-001; ng(n+1)=2.327634e+001;
n=47; farx(n+1)=6.619117e-001; foe(n+1)=2.475853e+000; krok(n+1)=7.847380e-001; ng(n+1)=2.250076e+001;
n=48; farx(n+1)=6.646653e-001; foe(n+1)=2.474366e+000; krok(n+1)=3.923690e-001; ng(n+1)=5.509124e+000;
n=49; farx(n+1)=6.611754e-001; foe(n+1)=2.472456e+000; krok(n+1)=1.334889e+000; ng(n+1)=5.697501e+000;
n=50; farx(n+1)=6.564776e-001; foe(n+1)=2.470801e+000; krok(n+1)=8.256204e-001; ng(n+1)=1.760424e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
