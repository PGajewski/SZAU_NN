%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.547239e+002; foe(n+1)=2.528892e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.760502e+002; foe(n+1)=1.748082e+002; krok(n+1)=5.100741e-004; ng(n+1)=4.914118e+002;
n=2; farx(n+1)=5.996080e+001; foe(n+1)=6.083577e+001; krok(n+1)=1.511671e-002; ng(n+1)=1.167261e+002;
n=3; farx(n+1)=5.557935e+001; foe(n+1)=5.585521e+001; krok(n+1)=7.273797e-004; ng(n+1)=2.716208e+002;
n=4; farx(n+1)=5.461140e+001; foe(n+1)=5.357740e+001; krok(n+1)=1.739067e-003; ng(n+1)=2.138670e+002;
n=5; farx(n+1)=2.263873e+001; foe(n+1)=4.146498e+001; krok(n+1)=2.617232e-002; ng(n+1)=7.439246e+001;
n=6; farx(n+1)=1.609174e+001; foe(n+1)=3.853382e+001; krok(n+1)=4.800362e-003; ng(n+1)=6.468163e+002;
n=7; farx(n+1)=1.911010e+001; foe(n+1)=3.509330e+001; krok(n+1)=2.118041e-002; ng(n+1)=9.328159e+002;
n=8; farx(n+1)=1.802952e+001; foe(n+1)=3.418060e+001; krok(n+1)=2.067140e-002; ng(n+1)=4.695182e+002;
n=9; farx(n+1)=1.386561e+001; foe(n+1)=3.133393e+001; krok(n+1)=1.069268e-002; ng(n+1)=4.603108e+002;
n=10; farx(n+1)=1.362738e+001; foe(n+1)=3.031435e+001; krok(n+1)=1.964862e-002; ng(n+1)=6.308246e+002;
n=11; farx(n+1)=1.323552e+001; foe(n+1)=2.738497e+001; krok(n+1)=8.268560e-002; ng(n+1)=6.431919e+002;
n=12; farx(n+1)=1.003628e+001; foe(n+1)=1.839365e+001; krok(n+1)=4.306047e-001; ng(n+1)=7.609467e+002;
n=13; farx(n+1)=8.030190e+000; foe(n+1)=1.552954e+001; krok(n+1)=6.385486e-002; ng(n+1)=3.434489e+002;
n=14; farx(n+1)=4.797768e+000; foe(n+1)=1.228617e+001; krok(n+1)=1.565625e-001; ng(n+1)=2.157189e+002;
n=15; farx(n+1)=3.462108e+000; foe(n+1)=1.130271e+001; krok(n+1)=3.691454e-001; ng(n+1)=3.937926e+002;
n=16; farx(n+1)=3.295512e+000; foe(n+1)=1.018296e+001; krok(n+1)=2.640676e-001; ng(n+1)=2.994779e+002;
n=17; farx(n+1)=1.794861e+000; foe(n+1)=8.924928e+000; krok(n+1)=3.505976e-001; ng(n+1)=6.193225e+001;
n=18; farx(n+1)=1.334875e+000; foe(n+1)=8.262653e+000; krok(n+1)=2.775732e-001; ng(n+1)=2.437376e+002;
n=19; farx(n+1)=1.021882e+000; foe(n+1)=7.748021e+000; krok(n+1)=4.065930e-001; ng(n+1)=3.078033e+002;
n=20; farx(n+1)=8.601028e-001; foe(n+1)=7.181238e+000; krok(n+1)=9.397835e-001; ng(n+1)=2.492634e+001;
n=21; farx(n+1)=9.089398e-001; foe(n+1)=6.861718e+000; krok(n+1)=8.885059e-001; ng(n+1)=9.730288e+001;
n=22; farx(n+1)=1.062937e+000; foe(n+1)=6.708871e+000; krok(n+1)=5.108389e-001; ng(n+1)=8.977027e+001;
n=23; farx(n+1)=1.153376e+000; foe(n+1)=6.537763e+000; krok(n+1)=6.710813e-001; ng(n+1)=1.262316e+002;
n=24; farx(n+1)=1.143390e+000; foe(n+1)=6.384223e+000; krok(n+1)=5.571345e-001; ng(n+1)=5.196637e+001;
n=25; farx(n+1)=1.050218e+000; foe(n+1)=5.892571e+000; krok(n+1)=1.663205e+000; ng(n+1)=7.946183e+001;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.046706e+000; foe(n+1)=5.852081e+000; krok(n+1)=1.416371e-005; ng(n+1)=1.552142e+002;
n=27; farx(n+1)=1.043510e+000; foe(n+1)=5.808853e+000; krok(n+1)=4.334515e-005; ng(n+1)=1.145419e+002;
n=28; farx(n+1)=1.013371e+000; foe(n+1)=5.742160e+000; krok(n+1)=4.770094e-004; ng(n+1)=4.039970e+001;
n=29; farx(n+1)=1.004606e+000; foe(n+1)=5.495061e+000; krok(n+1)=4.935931e-003; ng(n+1)=2.037050e+001;
n=30; farx(n+1)=9.597982e-001; foe(n+1)=4.785853e+000; krok(n+1)=2.173419e-002; ng(n+1)=2.507525e+001;
n=31; farx(n+1)=9.955236e-001; foe(n+1)=4.399305e+000; krok(n+1)=2.739044e-003; ng(n+1)=3.051523e+002;
n=32; farx(n+1)=1.046888e+000; foe(n+1)=4.196824e+000; krok(n+1)=5.767896e-003; ng(n+1)=1.840260e+002;
n=33; farx(n+1)=1.107079e+000; foe(n+1)=4.148428e+000; krok(n+1)=5.482403e-002; ng(n+1)=2.868823e+002;
n=34; farx(n+1)=1.291391e+000; foe(n+1)=3.969422e+000; krok(n+1)=5.455281e-002; ng(n+1)=2.888044e+002;
n=35; farx(n+1)=9.335803e-001; foe(n+1)=3.599720e+000; krok(n+1)=2.357265e-001; ng(n+1)=2.392516e+002;
n=36; farx(n+1)=7.798171e-001; foe(n+1)=3.056913e+000; krok(n+1)=1.484981e-001; ng(n+1)=3.156755e+002;
n=37; farx(n+1)=8.934867e-001; foe(n+1)=2.691391e+000; krok(n+1)=3.276650e-001; ng(n+1)=1.507819e+002;
n=38; farx(n+1)=7.934155e-001; foe(n+1)=2.161204e+000; krok(n+1)=1.443061e-001; ng(n+1)=3.878247e+001;
n=39; farx(n+1)=8.129535e-001; foe(n+1)=2.072669e+000; krok(n+1)=1.181740e-001; ng(n+1)=1.445533e+002;
n=40; farx(n+1)=6.232469e-001; foe(n+1)=1.874877e+000; krok(n+1)=3.768236e-001; ng(n+1)=1.104252e+002;
n=41; farx(n+1)=5.469548e-001; foe(n+1)=1.806253e+000; krok(n+1)=2.268451e-001; ng(n+1)=8.790676e+001;
n=42; farx(n+1)=6.149109e-001; foe(n+1)=1.678381e+000; krok(n+1)=1.104103e+000; ng(n+1)=9.022509e+001;
n=43; farx(n+1)=5.913373e-001; foe(n+1)=1.657215e+000; krok(n+1)=5.596492e-001; ng(n+1)=5.621902e+001;
n=44; farx(n+1)=5.585445e-001; foe(n+1)=1.631576e+000; krok(n+1)=7.296149e-001; ng(n+1)=5.549389e+001;
n=45; farx(n+1)=5.264320e-001; foe(n+1)=1.602128e+000; krok(n+1)=2.416435e+000; ng(n+1)=7.162711e+001;
n=46; farx(n+1)=5.221519e-001; foe(n+1)=1.590161e+000; krok(n+1)=1.269080e+000; ng(n+1)=2.116972e+001;
n=47; farx(n+1)=5.236852e-001; foe(n+1)=1.583946e+000; krok(n+1)=1.569476e+000; ng(n+1)=1.769378e+001;
n=48; farx(n+1)=5.164156e-001; foe(n+1)=1.580908e+000; krok(n+1)=4.747686e-001; ng(n+1)=2.406273e+001;
n=49; farx(n+1)=5.082393e-001; foe(n+1)=1.577799e+000; krok(n+1)=1.249330e+000; ng(n+1)=2.372378e+001;
n=50; farx(n+1)=5.192223e-001; foe(n+1)=1.572161e+000; krok(n+1)=3.887397e+000; ng(n+1)=6.633806e+000;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
