%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.479095e+002; foe(n+1)=2.549549e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.542314e+002; foe(n+1)=1.567421e+002; krok(n+1)=4.927090e-004; ng(n+1)=1.497600e+003;
n=2; farx(n+1)=5.952111e+001; foe(n+1)=5.857059e+001; krok(n+1)=4.427789e-003; ng(n+1)=6.782448e+002;
n=3; farx(n+1)=5.842147e+001; foe(n+1)=5.718371e+001; krok(n+1)=2.051078e-003; ng(n+1)=1.787178e+002;
n=4; farx(n+1)=5.016783e+001; foe(n+1)=5.565877e+001; krok(n+1)=1.086709e-002; ng(n+1)=7.637271e+001;
n=5; farx(n+1)=2.185764e+001; foe(n+1)=4.119065e+001; krok(n+1)=1.725161e-002; ng(n+1)=1.875578e+002;
n=6; farx(n+1)=1.454991e+001; foe(n+1)=3.623960e+001; krok(n+1)=2.487263e-003; ng(n+1)=5.023984e+002;
n=7; farx(n+1)=8.292027e+000; foe(n+1)=2.528120e+001; krok(n+1)=1.441974e-003; ng(n+1)=9.594913e+002;
n=8; farx(n+1)=8.049328e+000; foe(n+1)=2.322345e+001; krok(n+1)=7.432430e-004; ng(n+1)=2.036135e+003;
n=9; farx(n+1)=8.135956e+000; foe(n+1)=2.203939e+001; krok(n+1)=2.087321e-003; ng(n+1)=2.329280e+003;
n=10; farx(n+1)=8.449933e+000; foe(n+1)=2.048760e+001; krok(n+1)=2.707062e-003; ng(n+1)=2.308428e+003;
n=11; farx(n+1)=8.553175e+000; foe(n+1)=1.900726e+001; krok(n+1)=5.107684e-003; ng(n+1)=1.500268e+003;
n=12; farx(n+1)=9.061580e+000; foe(n+1)=1.746289e+001; krok(n+1)=1.434453e-002; ng(n+1)=6.861848e+002;
n=13; farx(n+1)=6.954083e+000; foe(n+1)=1.463264e+001; krok(n+1)=2.702970e-002; ng(n+1)=3.140935e+002;
n=14; farx(n+1)=5.878143e+000; foe(n+1)=1.270045e+001; krok(n+1)=2.900817e-003; ng(n+1)=7.210666e+002;
n=15; farx(n+1)=5.553265e+000; foe(n+1)=1.228969e+001; krok(n+1)=2.883948e-003; ng(n+1)=3.279549e+002;
n=16; farx(n+1)=3.124476e+000; foe(n+1)=8.894072e+000; krok(n+1)=2.284436e-002; ng(n+1)=2.594312e+002;
n=17; farx(n+1)=2.910660e+000; foe(n+1)=8.535928e+000; krok(n+1)=2.594055e-004; ng(n+1)=4.657902e+002;
n=18; farx(n+1)=2.541054e+000; foe(n+1)=7.883958e+000; krok(n+1)=2.209886e-003; ng(n+1)=6.575293e+002;
n=19; farx(n+1)=2.273692e+000; foe(n+1)=6.568322e+000; krok(n+1)=2.023809e-003; ng(n+1)=8.798301e+002;
n=20; farx(n+1)=2.208652e+000; foe(n+1)=6.190832e+000; krok(n+1)=3.481777e-003; ng(n+1)=7.346831e+002;
n=21; farx(n+1)=2.159065e+000; foe(n+1)=5.451476e+000; krok(n+1)=1.170476e-002; ng(n+1)=8.349064e+002;
n=22; farx(n+1)=2.176238e+000; foe(n+1)=5.049797e+000; krok(n+1)=8.368557e-003; ng(n+1)=5.051709e+002;
n=23; farx(n+1)=2.215272e+000; foe(n+1)=4.653757e+000; krok(n+1)=4.702595e-003; ng(n+1)=7.733507e+002;
n=24; farx(n+1)=1.998906e+000; foe(n+1)=4.265077e+000; krok(n+1)=7.860345e-003; ng(n+1)=3.267741e+002;
n=25; farx(n+1)=1.484842e+000; foe(n+1)=3.479442e+000; krok(n+1)=3.065020e-002; ng(n+1)=2.925452e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.486801e+000; foe(n+1)=3.428213e+000; krok(n+1)=2.829063e-005; ng(n+1)=1.913357e+002;
n=27; farx(n+1)=1.495401e+000; foe(n+1)=3.376947e+000; krok(n+1)=8.027267e-005; ng(n+1)=1.307681e+002;
n=28; farx(n+1)=1.457136e+000; foe(n+1)=3.348111e+000; krok(n+1)=1.278030e-004; ng(n+1)=7.192933e+001;
n=29; farx(n+1)=1.264824e+000; foe(n+1)=3.002165e+000; krok(n+1)=3.473809e-003; ng(n+1)=5.284854e+001;
n=30; farx(n+1)=9.316358e-001; foe(n+1)=2.539660e+000; krok(n+1)=1.647425e-002; ng(n+1)=5.482740e+001;
n=31; farx(n+1)=7.262871e-001; foe(n+1)=2.259565e+000; krok(n+1)=5.548179e-003; ng(n+1)=2.363039e+002;
n=32; farx(n+1)=6.027913e-001; foe(n+1)=1.956918e+000; krok(n+1)=4.900827e-003; ng(n+1)=4.307173e+002;
n=33; farx(n+1)=5.751361e-001; foe(n+1)=1.863417e+000; krok(n+1)=7.144846e-003; ng(n+1)=1.086188e+002;
n=34; farx(n+1)=5.654029e-001; foe(n+1)=1.713432e+000; krok(n+1)=1.696688e-002; ng(n+1)=1.106345e+002;
n=35; farx(n+1)=5.499305e-001; foe(n+1)=1.594564e+000; krok(n+1)=1.603572e-002; ng(n+1)=2.192219e+002;
n=36; farx(n+1)=5.093249e-001; foe(n+1)=1.397881e+000; krok(n+1)=2.667694e-002; ng(n+1)=1.553653e+002;
n=37; farx(n+1)=5.039730e-001; foe(n+1)=1.289877e+000; krok(n+1)=2.741623e-002; ng(n+1)=1.410195e+002;
n=38; farx(n+1)=5.011068e-001; foe(n+1)=1.221567e+000; krok(n+1)=8.003391e-003; ng(n+1)=2.062166e+002;
n=39; farx(n+1)=5.021644e-001; foe(n+1)=1.173118e+000; krok(n+1)=1.631060e-002; ng(n+1)=9.459458e+001;
n=40; farx(n+1)=4.997411e-001; foe(n+1)=1.110957e+000; krok(n+1)=4.095813e-002; ng(n+1)=1.087321e+002;
n=41; farx(n+1)=4.972235e-001; foe(n+1)=1.077271e+000; krok(n+1)=2.998718e-002; ng(n+1)=6.148781e+001;
n=42; farx(n+1)=5.107614e-001; foe(n+1)=1.012461e+000; krok(n+1)=5.184392e-002; ng(n+1)=1.135913e+002;
n=43; farx(n+1)=4.903121e-001; foe(n+1)=9.562541e-001; krok(n+1)=4.880471e-002; ng(n+1)=8.095430e+001;
n=44; farx(n+1)=4.647744e-001; foe(n+1)=9.101033e-001; krok(n+1)=1.012568e-002; ng(n+1)=1.516901e+002;
n=45; farx(n+1)=4.207952e-001; foe(n+1)=8.566234e-001; krok(n+1)=5.985461e-002; ng(n+1)=4.004840e+001;
n=46; farx(n+1)=3.758982e-001; foe(n+1)=7.998023e-001; krok(n+1)=2.219272e-002; ng(n+1)=1.311649e+002;
n=47; farx(n+1)=3.622451e-001; foe(n+1)=7.689665e-001; krok(n+1)=2.710756e-002; ng(n+1)=4.154560e+001;
n=48; farx(n+1)=3.552391e-001; foe(n+1)=7.434450e-001; krok(n+1)=6.418507e-002; ng(n+1)=1.055078e+002;
n=49; farx(n+1)=3.571037e-001; foe(n+1)=7.243651e-001; krok(n+1)=2.384691e-002; ng(n+1)=7.706462e+001;
n=50; farx(n+1)=3.652406e-001; foe(n+1)=6.590923e-001; krok(n+1)=2.044970e-001; ng(n+1)=7.593599e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
