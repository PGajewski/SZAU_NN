%uczenie predyktora oe
clear all;
n=0; farx(n+1)=1.984308e+002; foe(n+1)=1.941406e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.756602e+002; foe(n+1)=1.726556e+002; krok(n+1)=5.734420e-004; ng(n+1)=5.386992e+002;
n=2; farx(n+1)=5.978391e+001; foe(n+1)=6.310341e+001; krok(n+1)=1.038769e-002; ng(n+1)=3.046462e+002;
n=3; farx(n+1)=5.779948e+001; foe(n+1)=6.101852e+001; krok(n+1)=8.790130e-004; ng(n+1)=2.086986e+002;
n=4; farx(n+1)=3.008908e+001; foe(n+1)=5.623507e+001; krok(n+1)=2.447549e-002; ng(n+1)=5.230058e+001;
n=5; farx(n+1)=1.077185e+001; foe(n+1)=4.802879e+001; krok(n+1)=6.856719e-004; ng(n+1)=4.555403e+002;
n=6; farx(n+1)=7.554671e+000; foe(n+1)=4.613110e+001; krok(n+1)=3.369218e-004; ng(n+1)=1.774983e+003;
n=7; farx(n+1)=6.577886e+000; foe(n+1)=4.444293e+001; krok(n+1)=8.518466e-003; ng(n+1)=2.740040e+003;
n=8; farx(n+1)=5.433367e+000; foe(n+1)=3.189223e+001; krok(n+1)=3.215773e-003; ng(n+1)=3.285163e+003;
n=9; farx(n+1)=5.545974e+000; foe(n+1)=3.142470e+001; krok(n+1)=7.581373e-006; ng(n+1)=3.085024e+003;
n=10; farx(n+1)=6.955516e+000; foe(n+1)=2.743395e+001; krok(n+1)=2.025137e-002; ng(n+1)=2.984876e+003;
n=11; farx(n+1)=7.523330e+000; foe(n+1)=2.612989e+001; krok(n+1)=1.072587e-003; ng(n+1)=1.765127e+003;
n=12; farx(n+1)=8.081952e+000; foe(n+1)=2.539855e+001; krok(n+1)=3.113587e-003; ng(n+1)=1.114022e+003;
n=13; farx(n+1)=8.823627e+000; foe(n+1)=2.265818e+001; krok(n+1)=1.109636e-002; ng(n+1)=9.576577e+002;
n=14; farx(n+1)=6.788312e+000; foe(n+1)=1.991439e+001; krok(n+1)=3.535818e-002; ng(n+1)=2.556870e+002;
n=15; farx(n+1)=6.181020e+000; foe(n+1)=1.856993e+001; krok(n+1)=7.307512e-004; ng(n+1)=8.189746e+002;
n=16; farx(n+1)=5.942521e+000; foe(n+1)=1.755363e+001; krok(n+1)=3.367365e-003; ng(n+1)=2.719190e+002;
n=17; farx(n+1)=5.843400e+000; foe(n+1)=1.712794e+001; krok(n+1)=1.889294e-003; ng(n+1)=2.835206e+002;
n=18; farx(n+1)=4.654144e+000; foe(n+1)=1.582505e+001; krok(n+1)=9.602589e-003; ng(n+1)=3.067039e+002;
n=19; farx(n+1)=3.999986e+000; foe(n+1)=1.485935e+001; krok(n+1)=9.358895e-003; ng(n+1)=8.470591e+002;
n=20; farx(n+1)=3.634207e+000; foe(n+1)=1.382905e+001; krok(n+1)=1.131031e-002; ng(n+1)=6.892720e+002;
n=21; farx(n+1)=3.348639e+000; foe(n+1)=1.317572e+001; krok(n+1)=3.625910e-003; ng(n+1)=6.646917e+002;
n=22; farx(n+1)=1.815210e+000; foe(n+1)=9.379746e+000; krok(n+1)=2.565544e-002; ng(n+1)=3.225235e+002;
n=23; farx(n+1)=1.208686e+000; foe(n+1)=6.356356e+000; krok(n+1)=1.829287e-002; ng(n+1)=6.700599e+002;
n=24; farx(n+1)=1.150387e+000; foe(n+1)=5.831570e+000; krok(n+1)=1.219985e-002; ng(n+1)=5.672389e+002;
n=25; farx(n+1)=1.028922e+000; foe(n+1)=4.511204e+000; krok(n+1)=6.964181e-002; ng(n+1)=5.802996e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.026929e+000; foe(n+1)=4.451186e+000; krok(n+1)=6.123844e-006; ng(n+1)=2.915062e+002;
n=27; farx(n+1)=1.019292e+000; foe(n+1)=4.291303e+000; krok(n+1)=4.261938e-004; ng(n+1)=7.898337e+001;
n=28; farx(n+1)=9.927018e-001; foe(n+1)=3.749512e+000; krok(n+1)=6.158863e-005; ng(n+1)=3.868703e+002;
n=29; farx(n+1)=1.009088e+000; foe(n+1)=3.695657e+000; krok(n+1)=3.643472e-004; ng(n+1)=8.406245e+001;
n=30; farx(n+1)=1.003579e+000; foe(n+1)=3.001205e+000; krok(n+1)=4.689265e-003; ng(n+1)=6.138125e+001;
n=31; farx(n+1)=9.691445e-001; foe(n+1)=2.696872e+000; krok(n+1)=1.791379e-002; ng(n+1)=1.884899e+002;
n=32; farx(n+1)=9.571603e-001; foe(n+1)=2.566951e+000; krok(n+1)=4.150488e-003; ng(n+1)=3.032203e+002;
n=33; farx(n+1)=8.848614e-001; foe(n+1)=2.331427e+000; krok(n+1)=8.202462e-003; ng(n+1)=3.415106e+002;
n=34; farx(n+1)=8.749887e-001; foe(n+1)=2.142605e+000; krok(n+1)=1.638666e-002; ng(n+1)=1.345590e+002;
n=35; farx(n+1)=8.787807e-001; foe(n+1)=2.083319e+000; krok(n+1)=4.952046e-003; ng(n+1)=2.499226e+002;
n=36; farx(n+1)=9.099159e-001; foe(n+1)=1.882207e+000; krok(n+1)=7.860345e-003; ng(n+1)=3.653718e+002;
n=37; farx(n+1)=8.986664e-001; foe(n+1)=1.840307e+000; krok(n+1)=1.779884e-002; ng(n+1)=1.407161e+002;
n=38; farx(n+1)=8.723098e-001; foe(n+1)=1.675007e+000; krok(n+1)=5.405940e-002; ng(n+1)=1.742670e+002;
n=39; farx(n+1)=8.855839e-001; foe(n+1)=1.571356e+000; krok(n+1)=2.643052e-002; ng(n+1)=2.814658e+002;
n=40; farx(n+1)=8.557101e-001; foe(n+1)=1.535286e+000; krok(n+1)=2.618049e-002; ng(n+1)=1.210688e+002;
n=41; farx(n+1)=8.321278e-001; foe(n+1)=1.489578e+000; krok(n+1)=1.859267e-002; ng(n+1)=1.589116e+002;
n=42; farx(n+1)=7.835593e-001; foe(n+1)=1.426247e+000; krok(n+1)=3.781898e-002; ng(n+1)=1.076657e+002;
n=43; farx(n+1)=6.613298e-001; foe(n+1)=1.204358e+000; krok(n+1)=1.011122e-001; ng(n+1)=1.954729e+002;
n=44; farx(n+1)=6.080657e-001; foe(n+1)=1.079320e+000; krok(n+1)=6.900644e-002; ng(n+1)=3.623608e+002;
n=45; farx(n+1)=5.694998e-001; foe(n+1)=1.016047e+000; krok(n+1)=4.486711e-002; ng(n+1)=1.204846e+002;
n=46; farx(n+1)=5.458427e-001; foe(n+1)=9.700055e-001; krok(n+1)=4.017797e-002; ng(n+1)=7.853054e+001;
n=47; farx(n+1)=5.085190e-001; foe(n+1)=9.232423e-001; krok(n+1)=6.601690e-002; ng(n+1)=2.292522e+002;
n=48; farx(n+1)=5.169290e-001; foe(n+1)=8.467418e-001; krok(n+1)=1.961845e-001; ng(n+1)=1.415766e+002;
n=49; farx(n+1)=4.920808e-001; foe(n+1)=7.793083e-001; krok(n+1)=2.137961e-001; ng(n+1)=9.131035e+001;
n=50; farx(n+1)=4.467706e-001; foe(n+1)=7.202972e-001; krok(n+1)=2.688700e-001; ng(n+1)=9.252084e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
