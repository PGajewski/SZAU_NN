%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.817650e+002; foe(n+1)=2.820089e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.791174e+002; foe(n+1)=1.785291e+002; krok(n+1)=4.987505e-004; ng(n+1)=4.962018e+002;
n=2; farx(n+1)=6.909109e+001; foe(n+1)=6.448751e+001; krok(n+1)=1.643984e-001; ng(n+1)=1.048183e+001;
n=3; farx(n+1)=6.895004e+001; foe(n+1)=6.437154e+001; krok(n+1)=8.312662e-005; ng(n+1)=1.373647e+002;
n=4; farx(n+1)=6.663116e+001; foe(n+1)=6.163093e+001; krok(n+1)=5.793922e-002; ng(n+1)=1.458316e+002;
n=5; farx(n+1)=7.724291e+001; foe(n+1)=6.038901e+001; krok(n+1)=1.732520e-001; ng(n+1)=1.150917e+002;
n=6; farx(n+1)=5.095917e+001; foe(n+1)=5.039862e+001; krok(n+1)=2.180230e+000; ng(n+1)=5.721900e+001;
n=7; farx(n+1)=4.343051e+001; foe(n+1)=4.752315e+001; krok(n+1)=2.762929e-001; ng(n+1)=6.343294e+001;
n=8; farx(n+1)=2.904496e+001; foe(n+1)=4.079082e+001; krok(n+1)=8.726755e-001; ng(n+1)=4.620988e+001;
n=9; farx(n+1)=2.226855e+001; foe(n+1)=3.473550e+001; krok(n+1)=9.495372e-001; ng(n+1)=1.387434e+002;
n=10; farx(n+1)=2.156805e+001; foe(n+1)=3.447611e+001; krok(n+1)=1.082372e-001; ng(n+1)=6.969876e+001;
n=11; farx(n+1)=1.428447e+001; foe(n+1)=3.048681e+001; krok(n+1)=3.062035e+000; ng(n+1)=5.341232e+001;
n=12; farx(n+1)=1.417077e+001; foe(n+1)=3.043001e+001; krok(n+1)=1.095618e-002; ng(n+1)=1.511048e+002;
n=13; farx(n+1)=1.003960e+001; foe(n+1)=2.858748e+001; krok(n+1)=1.006208e-001; ng(n+1)=2.799141e+002;
n=14; farx(n+1)=9.467662e+000; foe(n+1)=2.842780e+001; krok(n+1)=2.191235e-002; ng(n+1)=8.293173e+001;
n=15; farx(n+1)=4.335413e+000; foe(n+1)=2.232725e+001; krok(n+1)=9.112329e-001; ng(n+1)=8.532908e+001;
n=16; farx(n+1)=4.190605e+000; foe(n+1)=2.136030e+001; krok(n+1)=5.335388e-002; ng(n+1)=4.980306e+002;
n=17; farx(n+1)=4.293858e+000; foe(n+1)=1.775532e+001; krok(n+1)=1.317940e-001; ng(n+1)=6.260792e+002;
n=18; farx(n+1)=4.225268e+000; foe(n+1)=1.765536e+001; krok(n+1)=1.830045e-002; ng(n+1)=2.646528e+002;
n=19; farx(n+1)=3.967134e+000; foe(n+1)=1.738466e+001; krok(n+1)=7.959957e-002; ng(n+1)=3.477163e+002;
n=20; farx(n+1)=3.285813e+000; foe(n+1)=1.528539e+001; krok(n+1)=1.044990e+000; ng(n+1)=2.275912e+002;
n=21; farx(n+1)=1.845176e+000; foe(n+1)=1.162236e+001; krok(n+1)=2.448603e+000; ng(n+1)=1.644019e+002;
n=22; farx(n+1)=1.746494e+000; foe(n+1)=1.144320e+001; krok(n+1)=1.010376e-001; ng(n+1)=1.488037e+002;
n=23; farx(n+1)=1.619010e+000; foe(n+1)=1.106769e+001; krok(n+1)=3.162969e-002; ng(n+1)=1.139064e+002;
n=24; farx(n+1)=1.324907e+000; foe(n+1)=1.020536e+001; krok(n+1)=1.098329e+000; ng(n+1)=1.484164e+002;
n=25; farx(n+1)=1.266403e+000; foe(n+1)=9.765001e+000; krok(n+1)=5.327344e-001; ng(n+1)=5.625275e+001;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.258858e+000; foe(n+1)=9.576737e+000; krok(n+1)=2.253084e-005; ng(n+1)=1.613625e+002;
n=27; farx(n+1)=1.212230e+000; foe(n+1)=9.509702e+000; krok(n+1)=1.864955e-004; ng(n+1)=3.448840e+001;
n=28; farx(n+1)=1.189092e+000; foe(n+1)=9.443504e+000; krok(n+1)=1.182363e-003; ng(n+1)=1.432215e+001;
n=29; farx(n+1)=1.180873e+000; foe(n+1)=9.419865e+000; krok(n+1)=2.707062e-003; ng(n+1)=1.115801e+001;
n=30; farx(n+1)=1.146156e+000; foe(n+1)=9.383113e+000; krok(n+1)=2.000182e-002; ng(n+1)=1.281971e+001;
n=31; farx(n+1)=1.150353e+000; foe(n+1)=9.197149e+000; krok(n+1)=3.450322e-002; ng(n+1)=3.066571e+001;
n=32; farx(n+1)=1.161170e+000; foe(n+1)=9.156190e+000; krok(n+1)=8.894860e-003; ng(n+1)=1.008463e+002;
n=33; farx(n+1)=1.147164e+000; foe(n+1)=9.010072e+000; krok(n+1)=1.459230e+000; ng(n+1)=7.206332e+001;
n=34; farx(n+1)=1.092774e+000; foe(n+1)=8.882980e+000; krok(n+1)=1.814761e+000; ng(n+1)=5.032792e+001;
n=35; farx(n+1)=1.077170e+000; foe(n+1)=8.860086e+000; krok(n+1)=6.426558e-001; ng(n+1)=3.228288e+001;
n=36; farx(n+1)=1.072161e+000; foe(n+1)=8.818340e+000; krok(n+1)=5.631821e-001; ng(n+1)=1.624525e+001;
n=37; farx(n+1)=1.063593e+000; foe(n+1)=8.802825e+000; krok(n+1)=5.683253e-001; ng(n+1)=3.482571e+001;
n=38; farx(n+1)=1.020204e+000; foe(n+1)=8.769841e+000; krok(n+1)=2.953163e+000; ng(n+1)=1.085468e+001;
n=39; farx(n+1)=1.003822e+000; foe(n+1)=8.755187e+000; krok(n+1)=4.213257e-001; ng(n+1)=2.595839e+001;
n=40; farx(n+1)=9.896314e-001; foe(n+1)=8.734024e+000; krok(n+1)=2.314697e+000; ng(n+1)=1.075455e+001;
n=41; farx(n+1)=9.773653e-001; foe(n+1)=8.693840e+000; krok(n+1)=1.625559e+000; ng(n+1)=6.782527e+000;
n=42; farx(n+1)=9.720062e-001; foe(n+1)=8.648984e+000; krok(n+1)=1.476581e+000; ng(n+1)=5.014095e+001;
n=43; farx(n+1)=9.828588e-001; foe(n+1)=8.542343e+000; krok(n+1)=3.289758e+000; ng(n+1)=2.857945e+001;
n=44; farx(n+1)=9.893447e-001; foe(n+1)=8.532692e+000; krok(n+1)=8.559000e-001; ng(n+1)=2.304615e+001;
n=45; farx(n+1)=9.936133e-001; foe(n+1)=8.529579e+000; krok(n+1)=4.747686e-001; ng(n+1)=1.695906e+001;
n=46; farx(n+1)=9.783173e-001; foe(n+1)=8.524519e+000; krok(n+1)=2.840668e+000; ng(n+1)=1.554469e+001;
n=47; farx(n+1)=9.772314e-001; foe(n+1)=8.522750e+000; krok(n+1)=2.055430e+000; ng(n+1)=4.227076e+000;
n=48; farx(n+1)=9.638086e-001; foe(n+1)=8.521311e+000; krok(n+1)=3.211979e+000; ng(n+1)=2.879728e+000;
n=49; farx(n+1)=9.264984e-001; foe(n+1)=8.510790e+000; krok(n+1)=1.138196e+001; ng(n+1)=3.769739e+000;
n=50; farx(n+1)=8.973381e-001; foe(n+1)=8.496360e+000; krok(n+1)=4.036727e+000; ng(n+1)=2.031858e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
