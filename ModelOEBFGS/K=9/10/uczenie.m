%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.343776e+002; foe(n+1)=2.299320e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.769995e+002; foe(n+1)=1.752120e+002; krok(n+1)=5.294487e-004; ng(n+1)=9.152509e+002;
n=2; farx(n+1)=6.111705e+001; foe(n+1)=6.256919e+001; krok(n+1)=1.042882e-002; ng(n+1)=3.845147e+002;
n=3; farx(n+1)=5.935894e+001; foe(n+1)=6.213139e+001; krok(n+1)=5.704577e-004; ng(n+1)=1.712753e+002;
n=4; farx(n+1)=3.049324e+001; foe(n+1)=5.730101e+001; krok(n+1)=1.759944e-002; ng(n+1)=8.208533e+001;
n=5; farx(n+1)=1.280550e+001; foe(n+1)=5.295789e+001; krok(n+1)=4.080480e-004; ng(n+1)=5.845188e+002;
n=6; farx(n+1)=5.384931e+000; foe(n+1)=4.882242e+001; krok(n+1)=7.632150e-003; ng(n+1)=1.515206e+003;
n=7; farx(n+1)=3.957154e+000; foe(n+1)=4.712179e+001; krok(n+1)=3.229906e-004; ng(n+1)=3.446449e+003;
n=8; farx(n+1)=3.460750e+000; foe(n+1)=4.602186e+001; krok(n+1)=5.157835e-004; ng(n+1)=4.875792e+003;
n=9; farx(n+1)=3.508256e+000; foe(n+1)=4.493311e+001; krok(n+1)=9.446471e-004; ng(n+1)=6.041155e+003;
n=10; farx(n+1)=3.615355e+000; foe(n+1)=4.450803e+001; krok(n+1)=1.516318e-003; ng(n+1)=5.778874e+003;
n=11; farx(n+1)=4.551569e+000; foe(n+1)=4.101452e+001; krok(n+1)=6.314847e-003; ng(n+1)=5.345250e+003;
n=12; farx(n+1)=4.879629e+000; foe(n+1)=4.043437e+001; krok(n+1)=4.015818e-004; ng(n+1)=3.663129e+003;
n=13; farx(n+1)=5.848662e+000; foe(n+1)=3.919614e+001; krok(n+1)=2.716773e-003; ng(n+1)=3.269815e+003;
n=14; farx(n+1)=6.338286e+000; foe(n+1)=3.851859e+001; krok(n+1)=1.389400e-003; ng(n+1)=3.147180e+003;
n=15; farx(n+1)=7.991367e+000; foe(n+1)=3.541461e+001; krok(n+1)=2.488900e-003; ng(n+1)=2.600432e+003;
n=16; farx(n+1)=1.070706e+001; foe(n+1)=2.933806e+001; krok(n+1)=7.251821e-003; ng(n+1)=2.755671e+003;
n=17; farx(n+1)=1.119718e+001; foe(n+1)=2.746231e+001; krok(n+1)=2.032216e-003; ng(n+1)=5.399980e+002;
n=18; farx(n+1)=1.185339e+001; foe(n+1)=2.542219e+001; krok(n+1)=1.532691e-003; ng(n+1)=8.562294e+002;
n=19; farx(n+1)=1.181803e+001; foe(n+1)=2.494197e+001; krok(n+1)=1.299476e-003; ng(n+1)=4.737631e+002;
n=20; farx(n+1)=9.817765e+000; foe(n+1)=2.011237e+001; krok(n+1)=1.999069e-002; ng(n+1)=5.851599e+002;
n=21; farx(n+1)=8.890166e+000; foe(n+1)=1.783432e+001; krok(n+1)=3.621201e-003; ng(n+1)=8.468440e+002;
n=22; farx(n+1)=8.218543e+000; foe(n+1)=1.653301e+001; krok(n+1)=2.707062e-003; ng(n+1)=8.164375e+002;
n=23; farx(n+1)=7.741121e+000; foe(n+1)=1.524717e+001; krok(n+1)=1.516318e-003; ng(n+1)=1.174880e+003;
n=24; farx(n+1)=6.497382e+000; foe(n+1)=1.240111e+001; krok(n+1)=5.414124e-003; ng(n+1)=4.605206e+002;
n=25; farx(n+1)=5.169082e+000; foe(n+1)=9.749148e+000; krok(n+1)=6.764823e-003; ng(n+1)=9.450474e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=5.141981e+000; foe(n+1)=9.640674e+000; krok(n+1)=1.747813e-005; ng(n+1)=3.087219e+002;
n=27; farx(n+1)=5.149169e+000; foe(n+1)=9.172240e+000; krok(n+1)=1.297027e-004; ng(n+1)=2.752307e+002;
n=28; farx(n+1)=5.107411e+000; foe(n+1)=9.091355e+000; krok(n+1)=6.447294e-005; ng(n+1)=1.509005e+002;
n=29; farx(n+1)=3.828292e+000; foe(n+1)=7.037458e+000; krok(n+1)=5.915528e-003; ng(n+1)=1.278287e+002;
n=30; farx(n+1)=2.543695e+000; foe(n+1)=4.323482e+000; krok(n+1)=8.029022e-003; ng(n+1)=9.564315e+002;
n=31; farx(n+1)=2.425666e+000; foe(n+1)=4.141300e+000; krok(n+1)=2.677949e-003; ng(n+1)=3.798321e+002;
n=32; farx(n+1)=2.129836e+000; foe(n+1)=3.540099e+000; krok(n+1)=7.632150e-003; ng(n+1)=1.148729e+002;
n=33; farx(n+1)=1.851871e+000; foe(n+1)=3.015852e+000; krok(n+1)=1.394089e-002; ng(n+1)=5.217671e+002;
n=34; farx(n+1)=1.751542e+000; foe(n+1)=2.785792e+000; krok(n+1)=1.564230e-003; ng(n+1)=5.668299e+002;
n=35; farx(n+1)=1.504720e+000; foe(n+1)=2.338034e+000; krok(n+1)=1.725161e-002; ng(n+1)=3.076111e+002;
n=36; farx(n+1)=1.356396e+000; foe(n+1)=2.093557e+000; krok(n+1)=9.248075e-003; ng(n+1)=3.578451e+002;
n=37; farx(n+1)=1.278312e+000; foe(n+1)=2.004235e+000; krok(n+1)=5.829556e-003; ng(n+1)=3.220355e+002;
n=38; farx(n+1)=1.057879e+000; foe(n+1)=1.817415e+000; krok(n+1)=1.706813e-002; ng(n+1)=1.284289e+002;
n=39; farx(n+1)=7.967671e-001; foe(n+1)=1.462594e+000; krok(n+1)=3.104538e-002; ng(n+1)=3.107684e+002;
n=40; farx(n+1)=7.348962e-001; foe(n+1)=1.301161e+000; krok(n+1)=6.822948e-003; ng(n+1)=2.370576e+002;
n=41; farx(n+1)=6.973979e-001; foe(n+1)=1.176041e+000; krok(n+1)=1.275465e-002; ng(n+1)=1.432155e+002;
n=42; farx(n+1)=6.706744e-001; foe(n+1)=1.132662e+000; krok(n+1)=1.958773e-002; ng(n+1)=1.942785e+002;
n=43; farx(n+1)=6.018900e-001; foe(n+1)=1.077979e+000; krok(n+1)=2.900728e-002; ng(n+1)=4.773645e+001;
n=44; farx(n+1)=5.537219e-001; foe(n+1)=1.039444e+000; krok(n+1)=1.982937e-002; ng(n+1)=6.577094e+001;
n=45; farx(n+1)=5.338098e-001; foe(n+1)=1.017141e+000; krok(n+1)=1.275465e-002; ng(n+1)=1.165947e+002;
n=46; farx(n+1)=5.064491e-001; foe(n+1)=9.781586e-001; krok(n+1)=6.918848e-002; ng(n+1)=5.127365e+001;
n=47; farx(n+1)=4.956905e-001; foe(n+1)=9.403273e-001; krok(n+1)=6.101054e-002; ng(n+1)=5.321470e+001;
n=48; farx(n+1)=4.935540e-001; foe(n+1)=9.304859e-001; krok(n+1)=2.128042e-002; ng(n+1)=1.090002e+002;
n=49; farx(n+1)=4.887556e-001; foe(n+1)=8.751007e-001; krok(n+1)=1.732520e-001; ng(n+1)=3.616002e+001;
n=50; farx(n+1)=4.879459e-001; foe(n+1)=8.521706e-001; krok(n+1)=1.989989e-002; ng(n+1)=9.530044e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
