%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.769252e+002; foe(n+1)=2.752512e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.669274e+002; foe(n+1)=1.657617e+002; krok(n+1)=4.957365e-004; ng(n+1)=8.055843e+002;
n=2; farx(n+1)=6.762969e+001; foe(n+1)=6.877965e+001; krok(n+1)=9.555317e-003; ng(n+1)=1.997925e+002;
n=3; farx(n+1)=6.413538e+001; foe(n+1)=6.387376e+001; krok(n+1)=1.555872e-003; ng(n+1)=2.585709e+002;
n=4; farx(n+1)=6.456232e+001; foe(n+1)=6.176296e+001; krok(n+1)=9.955600e-003; ng(n+1)=1.784774e+002;
n=5; farx(n+1)=4.495505e+001; foe(n+1)=5.810496e+001; krok(n+1)=2.594061e-002; ng(n+1)=3.364104e+001;
n=6; farx(n+1)=2.033899e+001; foe(n+1)=5.132686e+001; krok(n+1)=1.450364e-002; ng(n+1)=2.403267e+002;
n=7; farx(n+1)=1.676721e+001; foe(n+1)=5.074978e+001; krok(n+1)=8.489916e-005; ng(n+1)=6.572554e+002;
n=8; farx(n+1)=1.268048e+001; foe(n+1)=4.920486e+001; krok(n+1)=2.607440e-002; ng(n+1)=8.461631e+002;
n=9; farx(n+1)=9.346043e+000; foe(n+1)=4.484871e+001; krok(n+1)=4.290347e-003; ng(n+1)=1.167286e+003;
n=10; farx(n+1)=8.812987e+000; foe(n+1)=4.409055e+001; krok(n+1)=3.927263e-004; ng(n+1)=1.450649e+003;
n=11; farx(n+1)=8.575084e+000; foe(n+1)=4.158780e+001; krok(n+1)=1.611092e-002; ng(n+1)=1.914921e+003;
n=12; farx(n+1)=8.010330e+000; foe(n+1)=3.224723e+001; krok(n+1)=3.899500e-002; ng(n+1)=2.190543e+003;
n=13; farx(n+1)=8.059617e+000; foe(n+1)=3.157677e+001; krok(n+1)=7.010405e-004; ng(n+1)=1.383712e+003;
n=14; farx(n+1)=8.590271e+000; foe(n+1)=2.897169e+001; krok(n+1)=1.226153e-002; ng(n+1)=1.553815e+003;
n=15; farx(n+1)=6.721892e+000; foe(n+1)=1.899422e+001; krok(n+1)=6.384007e-002; ng(n+1)=1.156412e+003;
n=16; farx(n+1)=6.217018e+000; foe(n+1)=1.719623e+001; krok(n+1)=7.280720e-002; ng(n+1)=6.487545e+002;
n=17; farx(n+1)=4.819266e+000; foe(n+1)=1.335641e+001; krok(n+1)=4.187571e-001; ng(n+1)=2.648945e+002;
n=18; farx(n+1)=4.011882e+000; foe(n+1)=1.131435e+001; krok(n+1)=1.047148e-001; ng(n+1)=2.186312e+002;
n=19; farx(n+1)=3.576089e+000; foe(n+1)=9.952715e+000; krok(n+1)=3.020544e-001; ng(n+1)=3.243498e+002;
n=20; farx(n+1)=3.238347e+000; foe(n+1)=8.990602e+000; krok(n+1)=2.905057e-001; ng(n+1)=3.963663e+002;
n=21; farx(n+1)=2.530540e+000; foe(n+1)=7.911178e+000; krok(n+1)=6.236201e-001; ng(n+1)=2.295947e+002;
n=22; farx(n+1)=2.439031e+000; foe(n+1)=7.571276e+000; krok(n+1)=3.240219e-001; ng(n+1)=2.287322e+002;
n=23; farx(n+1)=2.284615e+000; foe(n+1)=7.123217e+000; krok(n+1)=7.926233e-001; ng(n+1)=9.420790e+001;
n=24; farx(n+1)=2.263296e+000; foe(n+1)=6.678108e+000; krok(n+1)=1.322970e+000; ng(n+1)=1.673389e+002;
n=25; farx(n+1)=1.944239e+000; foe(n+1)=6.050446e+000; krok(n+1)=1.324463e+000; ng(n+1)=1.970192e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.924889e+000; foe(n+1)=5.877558e+000; krok(n+1)=3.242569e-005; ng(n+1)=2.340026e+002;
n=27; farx(n+1)=1.930977e+000; foe(n+1)=5.770844e+000; krok(n+1)=2.543436e-005; ng(n+1)=2.467404e+002;
n=28; farx(n+1)=1.879384e+000; foe(n+1)=5.707621e+000; krok(n+1)=1.547514e-004; ng(n+1)=4.868169e+001;
n=29; farx(n+1)=1.544258e+000; foe(n+1)=4.870246e+000; krok(n+1)=2.047906e-002; ng(n+1)=2.309078e+001;
n=30; farx(n+1)=1.477422e+000; foe(n+1)=4.500414e+000; krok(n+1)=3.195265e-003; ng(n+1)=1.389405e+002;
n=31; farx(n+1)=1.438551e+000; foe(n+1)=4.331520e+000; krok(n+1)=3.419137e-003; ng(n+1)=2.362371e+002;
n=32; farx(n+1)=1.351971e+000; foe(n+1)=3.931654e+000; krok(n+1)=1.082825e-002; ng(n+1)=3.222161e+002;
n=33; farx(n+1)=1.256312e+000; foe(n+1)=3.770748e+000; krok(n+1)=3.039725e-002; ng(n+1)=1.796277e+002;
n=34; farx(n+1)=1.189257e+000; foe(n+1)=3.605937e+000; krok(n+1)=3.571281e-002; ng(n+1)=1.709431e+002;
n=35; farx(n+1)=1.153170e+000; foe(n+1)=3.468177e+000; krok(n+1)=8.744518e-003; ng(n+1)=1.002497e+002;
n=36; farx(n+1)=1.088185e+000; foe(n+1)=3.379844e+000; krok(n+1)=5.018717e-002; ng(n+1)=4.940464e+001;
n=37; farx(n+1)=1.107104e+000; foe(n+1)=2.902753e+000; krok(n+1)=1.478545e-001; ng(n+1)=1.458691e+002;
n=38; farx(n+1)=1.129978e+000; foe(n+1)=2.817408e+000; krok(n+1)=1.105941e-001; ng(n+1)=1.020533e+002;
n=39; farx(n+1)=9.666371e-001; foe(n+1)=2.408046e+000; krok(n+1)=2.696528e-001; ng(n+1)=5.606711e+001;
n=40; farx(n+1)=8.693274e-001; foe(n+1)=2.106174e+000; krok(n+1)=5.235795e-001; ng(n+1)=8.784889e+001;
n=41; farx(n+1)=8.239963e-001; foe(n+1)=1.955545e+000; krok(n+1)=5.804224e-001; ng(n+1)=5.043498e+001;
n=42; farx(n+1)=7.850481e-001; foe(n+1)=1.695572e+000; krok(n+1)=3.112246e-001; ng(n+1)=1.119324e+002;
n=43; farx(n+1)=7.661193e-001; foe(n+1)=1.639524e+000; krok(n+1)=2.155114e-001; ng(n+1)=3.507650e+001;
n=44; farx(n+1)=6.624767e-001; foe(n+1)=1.523939e+000; krok(n+1)=3.828290e-001; ng(n+1)=8.208519e+001;
n=45; farx(n+1)=6.293725e-001; foe(n+1)=1.492086e+000; krok(n+1)=5.431439e-001; ng(n+1)=6.559219e+001;
n=46; farx(n+1)=6.228899e-001; foe(n+1)=1.457541e+000; krok(n+1)=5.856145e-001; ng(n+1)=2.143197e+001;
n=47; farx(n+1)=6.893437e-001; foe(n+1)=1.396715e+000; krok(n+1)=4.144507e-001; ng(n+1)=3.760017e+001;
n=48; farx(n+1)=6.218753e-001; foe(n+1)=1.345665e+000; krok(n+1)=1.238762e+000; ng(n+1)=2.629980e+001;
n=49; farx(n+1)=5.888343e-001; foe(n+1)=1.312039e+000; krok(n+1)=8.316025e-001; ng(n+1)=3.030215e+001;
n=50; farx(n+1)=5.716636e-001; foe(n+1)=1.293622e+000; krok(n+1)=6.428250e-001; ng(n+1)=2.615625e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
