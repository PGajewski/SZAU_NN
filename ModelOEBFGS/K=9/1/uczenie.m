%uczenie predyktora oe
clear all;
n=0; farx(n+1)=1.913137e+002; foe(n+1)=1.933879e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.604687e+002; foe(n+1)=1.621401e+002; krok(n+1)=5.304139e-004; ng(n+1)=7.998576e+002;
n=2; farx(n+1)=6.832734e+001; foe(n+1)=6.840209e+001; krok(n+1)=6.130766e-003; ng(n+1)=4.567688e+002;
n=3; farx(n+1)=6.427650e+001; foe(n+1)=6.282115e+001; krok(n+1)=2.023175e-003; ng(n+1)=3.595902e+002;
n=4; farx(n+1)=5.459035e+001; foe(n+1)=6.113935e+001; krok(n+1)=1.778267e-002; ng(n+1)=6.339256e+001;
n=5; farx(n+1)=2.313187e+001; foe(n+1)=5.356098e+001; krok(n+1)=7.852675e-003; ng(n+1)=2.178621e+002;
n=6; farx(n+1)=7.678089e+000; foe(n+1)=4.420901e+001; krok(n+1)=3.401267e-003; ng(n+1)=9.716332e+002;
n=7; farx(n+1)=6.319558e+000; foe(n+1)=4.300349e+001; krok(n+1)=7.103937e-005; ng(n+1)=2.105286e+003;
n=8; farx(n+1)=5.752093e+000; foe(n+1)=4.187057e+001; krok(n+1)=2.351298e-003; ng(n+1)=2.777325e+003;
n=9; farx(n+1)=5.713347e+000; foe(n+1)=3.956451e+001; krok(n+1)=7.383609e-004; ng(n+1)=3.257990e+003;
n=10; farx(n+1)=5.874970e+000; foe(n+1)=3.862807e+001; krok(n+1)=2.634602e-003; ng(n+1)=4.305421e+003;
n=11; farx(n+1)=5.079260e+000; foe(n+1)=2.886933e+001; krok(n+1)=8.213360e-003; ng(n+1)=4.142794e+003;
n=12; farx(n+1)=5.569718e+000; foe(n+1)=2.681008e+001; krok(n+1)=4.526501e-004; ng(n+1)=1.742948e+003;
n=13; farx(n+1)=6.822932e+000; foe(n+1)=2.368529e+001; krok(n+1)=6.970444e-003; ng(n+1)=1.128356e+003;
n=14; farx(n+1)=7.043752e+000; foe(n+1)=2.165731e+001; krok(n+1)=5.711090e-003; ng(n+1)=3.902242e+002;
n=15; farx(n+1)=5.130734e+000; foe(n+1)=1.949499e+001; krok(n+1)=9.495353e-003; ng(n+1)=2.397570e+002;
n=16; farx(n+1)=2.607578e+000; foe(n+1)=1.408018e+001; krok(n+1)=2.428721e-002; ng(n+1)=3.524328e+002;
n=17; farx(n+1)=2.380743e+000; foe(n+1)=1.336101e+001; krok(n+1)=8.006062e-004; ng(n+1)=4.617324e+002;
n=18; farx(n+1)=2.763124e+000; foe(n+1)=1.170968e+001; krok(n+1)=1.276026e-002; ng(n+1)=4.775975e+002;
n=19; farx(n+1)=2.662724e+000; foe(n+1)=1.042913e+001; krok(n+1)=2.176307e-003; ng(n+1)=5.395897e+002;
n=20; farx(n+1)=1.939574e+000; foe(n+1)=7.585017e+000; krok(n+1)=2.660052e-003; ng(n+1)=6.255191e+002;
n=21; farx(n+1)=1.775765e+000; foe(n+1)=6.525859e+000; krok(n+1)=7.609550e-003; ng(n+1)=5.448285e+002;
n=22; farx(n+1)=1.813518e+000; foe(n+1)=5.752795e+000; krok(n+1)=7.099884e-003; ng(n+1)=3.890262e+002;
n=23; farx(n+1)=1.626677e+000; foe(n+1)=4.671540e+000; krok(n+1)=9.358895e-003; ng(n+1)=1.827337e+002;
n=24; farx(n+1)=1.624756e+000; foe(n+1)=4.628845e+000; krok(n+1)=1.593673e-003; ng(n+1)=2.504646e+002;
n=25; farx(n+1)=1.373035e+000; foe(n+1)=4.065411e+000; krok(n+1)=2.691279e-002; ng(n+1)=2.896356e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.365456e+000; foe(n+1)=4.039747e+000; krok(n+1)=1.763713e-005; ng(n+1)=1.744327e+002;
n=27; farx(n+1)=1.408114e+000; foe(n+1)=3.875822e+000; krok(n+1)=2.554609e-004; ng(n+1)=1.053322e+002;
n=28; farx(n+1)=1.257567e+000; foe(n+1)=3.422397e+000; krok(n+1)=3.984182e-004; ng(n+1)=1.516065e+002;
n=29; farx(n+1)=1.227379e+000; foe(n+1)=3.361512e+000; krok(n+1)=1.264385e-004; ng(n+1)=1.009760e+002;
n=30; farx(n+1)=1.110321e+000; foe(n+1)=3.189239e+000; krok(n+1)=2.272397e-002; ng(n+1)=4.670697e+001;
n=31; farx(n+1)=9.421134e-001; foe(n+1)=2.808167e+000; krok(n+1)=2.351765e-002; ng(n+1)=5.307565e+001;
n=32; farx(n+1)=8.796552e-001; foe(n+1)=2.632476e+000; krok(n+1)=3.259300e-003; ng(n+1)=3.510536e+002;
n=33; farx(n+1)=8.051220e-001; foe(n+1)=2.406603e+000; krok(n+1)=2.097129e-002; ng(n+1)=1.217465e+002;
n=34; farx(n+1)=7.715466e-001; foe(n+1)=2.286924e+000; krok(n+1)=1.019447e-002; ng(n+1)=2.039823e+002;
n=35; farx(n+1)=7.404834e-001; foe(n+1)=2.094401e+000; krok(n+1)=2.502130e-002; ng(n+1)=2.629786e+002;
n=36; farx(n+1)=6.527125e-001; foe(n+1)=1.896978e+000; krok(n+1)=2.165649e-002; ng(n+1)=3.342736e+001;
n=37; farx(n+1)=5.785115e-001; foe(n+1)=1.647913e+000; krok(n+1)=6.853004e-003; ng(n+1)=3.103682e+002;
n=38; farx(n+1)=5.485905e-001; foe(n+1)=1.509649e+000; krok(n+1)=6.800237e-003; ng(n+1)=1.963065e+002;
n=39; farx(n+1)=5.152339e-001; foe(n+1)=1.395488e+000; krok(n+1)=2.611507e-002; ng(n+1)=1.976365e+002;
n=40; farx(n+1)=5.140571e-001; foe(n+1)=1.332080e+000; krok(n+1)=1.095618e-002; ng(n+1)=1.743925e+002;
n=41; farx(n+1)=5.123462e-001; foe(n+1)=1.286695e+000; krok(n+1)=1.262969e-002; ng(n+1)=1.030488e+002;
n=42; farx(n+1)=5.125049e-001; foe(n+1)=1.173883e+000; krok(n+1)=1.462925e-002; ng(n+1)=1.756259e+002;
n=43; farx(n+1)=5.191630e-001; foe(n+1)=1.137138e+000; krok(n+1)=2.399905e-002; ng(n+1)=1.891565e+002;
n=44; farx(n+1)=5.138880e-001; foe(n+1)=1.093052e+000; krok(n+1)=6.237096e-002; ng(n+1)=7.146334e+001;
n=45; farx(n+1)=5.144831e-001; foe(n+1)=1.064562e+000; krok(n+1)=3.450322e-002; ng(n+1)=1.222883e+002;
n=46; farx(n+1)=5.041198e-001; foe(n+1)=1.034451e+000; krok(n+1)=1.346726e-001; ng(n+1)=4.157021e+001;
n=47; farx(n+1)=4.901387e-001; foe(n+1)=1.019753e+000; krok(n+1)=4.438543e-002; ng(n+1)=5.050341e+001;
n=48; farx(n+1)=4.917392e-001; foe(n+1)=9.815638e-001; krok(n+1)=9.233170e-002; ng(n+1)=5.359501e+001;
n=49; farx(n+1)=4.856608e-001; foe(n+1)=9.519664e-001; krok(n+1)=7.165516e-002; ng(n+1)=5.924878e+001;
n=50; farx(n+1)=4.822788e-001; foe(n+1)=9.420159e-001; krok(n+1)=1.321526e-002; ng(n+1)=8.032837e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
