%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.008252e+002; foe(n+1)=2.084744e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.694890e+002; foe(n+1)=1.766989e+002; krok(n+1)=5.033545e-004; ng(n+1)=6.447131e+002;
n=2; farx(n+1)=6.567952e+001; foe(n+1)=6.198087e+001; krok(n+1)=1.634950e-002; ng(n+1)=3.348582e+002;
n=3; farx(n+1)=6.094868e+001; foe(n+1)=6.144705e+001; krok(n+1)=2.041615e-003; ng(n+1)=9.399249e+001;
n=4; farx(n+1)=1.494257e+001; foe(n+1)=4.934609e+001; krok(n+1)=5.548179e-003; ng(n+1)=2.036267e+002;
n=5; farx(n+1)=9.706693e+000; foe(n+1)=4.769775e+001; krok(n+1)=3.867010e-005; ng(n+1)=1.278347e+003;
n=6; farx(n+1)=7.744718e+000; foe(n+1)=3.246830e+001; krok(n+1)=2.835564e-002; ng(n+1)=2.057253e+003;
n=7; farx(n+1)=7.948534e+000; foe(n+1)=3.188486e+001; krok(n+1)=4.229784e-005; ng(n+1)=2.600275e+003;
n=8; farx(n+1)=8.412018e+000; foe(n+1)=2.751396e+001; krok(n+1)=3.214905e-003; ng(n+1)=3.033354e+003;
n=9; farx(n+1)=8.377438e+000; foe(n+1)=2.619892e+001; krok(n+1)=1.636169e-003; ng(n+1)=1.873440e+003;
n=10; farx(n+1)=9.061659e+000; foe(n+1)=2.440140e+001; krok(n+1)=1.160327e-002; ng(n+1)=1.093003e+003;
n=11; farx(n+1)=9.209241e+000; foe(n+1)=2.340487e+001; krok(n+1)=3.910756e-003; ng(n+1)=5.076317e+002;
n=12; farx(n+1)=9.363783e+000; foe(n+1)=2.245461e+001; krok(n+1)=5.167850e-003; ng(n+1)=5.139387e+002;
n=13; farx(n+1)=8.849807e+000; foe(n+1)=2.079337e+001; krok(n+1)=8.855578e-003; ng(n+1)=6.455599e+002;
n=14; farx(n+1)=8.523792e+000; foe(n+1)=2.026828e+001; krok(n+1)=3.765398e-003; ng(n+1)=3.892696e+002;
n=15; farx(n+1)=6.720197e+000; foe(n+1)=1.872710e+001; krok(n+1)=2.691279e-002; ng(n+1)=3.225655e+002;
n=16; farx(n+1)=5.006045e+000; foe(n+1)=1.658302e+001; krok(n+1)=4.692985e-003; ng(n+1)=6.566134e+002;
n=17; farx(n+1)=4.131678e+000; foe(n+1)=1.533263e+001; krok(n+1)=1.524981e-003; ng(n+1)=5.466406e+002;
n=18; farx(n+1)=2.843075e+000; foe(n+1)=1.270298e+001; krok(n+1)=4.539517e-003; ng(n+1)=9.094351e+002;
n=19; farx(n+1)=2.380359e+000; foe(n+1)=1.151200e+001; krok(n+1)=1.078226e-003; ng(n+1)=6.510941e+002;
n=20; farx(n+1)=2.200428e+000; foe(n+1)=1.046448e+001; krok(n+1)=1.462416e-003; ng(n+1)=4.652588e+002;
n=21; farx(n+1)=1.890456e+000; foe(n+1)=6.607262e+000; krok(n+1)=1.574703e-002; ng(n+1)=6.132560e+002;
n=22; farx(n+1)=1.797968e+000; foe(n+1)=5.851417e+000; krok(n+1)=8.178849e-004; ng(n+1)=1.294320e+003;
n=23; farx(n+1)=1.658191e+000; foe(n+1)=4.702629e+000; krok(n+1)=3.378712e-003; ng(n+1)=6.730643e+002;
n=24; farx(n+1)=1.640012e+000; foe(n+1)=4.501593e+000; krok(n+1)=4.352152e-003; ng(n+1)=8.273424e+002;
n=25; farx(n+1)=1.358692e+000; foe(n+1)=3.755094e+000; krok(n+1)=2.142882e-002; ng(n+1)=5.794403e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.355040e+000; foe(n+1)=3.489990e+000; krok(n+1)=1.038706e-005; ng(n+1)=6.902917e+002;
n=27; farx(n+1)=1.349852e+000; foe(n+1)=3.456391e+000; krok(n+1)=7.938342e-006; ng(n+1)=2.512998e+002;
n=28; farx(n+1)=1.313832e+000; foe(n+1)=3.325479e+000; krok(n+1)=1.714180e-004; ng(n+1)=1.254046e+002;
n=29; farx(n+1)=1.340070e+000; foe(n+1)=3.074974e+000; krok(n+1)=6.592938e-004; ng(n+1)=9.138971e+001;
n=30; farx(n+1)=1.309452e+000; foe(n+1)=2.896264e+000; krok(n+1)=2.458210e-003; ng(n+1)=4.617264e+001;
n=31; farx(n+1)=1.135795e+000; foe(n+1)=2.547023e+000; krok(n+1)=6.288803e-003; ng(n+1)=4.343127e+001;
n=32; farx(n+1)=1.177589e+000; foe(n+1)=2.146213e+000; krok(n+1)=4.134280e-002; ng(n+1)=3.252076e+002;
n=33; farx(n+1)=1.169006e+000; foe(n+1)=2.124502e+000; krok(n+1)=2.013418e-003; ng(n+1)=1.900407e+002;
n=34; farx(n+1)=1.080140e+000; foe(n+1)=1.986970e+000; krok(n+1)=2.223039e-002; ng(n+1)=1.578173e+002;
n=35; farx(n+1)=9.734779e-001; foe(n+1)=1.851504e+000; krok(n+1)=1.208515e-002; ng(n+1)=1.721276e+002;
n=36; farx(n+1)=8.696475e-001; foe(n+1)=1.701978e+000; krok(n+1)=3.516052e-003; ng(n+1)=4.371438e+002;
n=37; farx(n+1)=8.009378e-001; foe(n+1)=1.634593e+000; krok(n+1)=8.053672e-003; ng(n+1)=1.004057e+002;
n=38; farx(n+1)=7.677866e-001; foe(n+1)=1.594771e+000; krok(n+1)=2.779047e-002; ng(n+1)=8.905952e+001;
n=39; farx(n+1)=7.274213e-001; foe(n+1)=1.493356e+000; krok(n+1)=1.843505e-002; ng(n+1)=9.633758e+001;
n=40; farx(n+1)=7.026933e-001; foe(n+1)=1.451870e+000; krok(n+1)=8.009570e-003; ng(n+1)=1.951523e+002;
n=41; farx(n+1)=6.721225e-001; foe(n+1)=1.301912e+000; krok(n+1)=5.801457e-002; ng(n+1)=2.489772e+002;
n=42; farx(n+1)=6.627421e-001; foe(n+1)=1.249077e+000; krok(n+1)=1.427439e-002; ng(n+1)=2.901606e+002;
n=43; farx(n+1)=6.396567e-001; foe(n+1)=1.193763e+000; krok(n+1)=5.558094e-002; ng(n+1)=8.771492e+001;
n=44; farx(n+1)=6.062787e-001; foe(n+1)=1.022402e+000; krok(n+1)=7.485385e-002; ng(n+1)=2.417780e+002;
n=45; farx(n+1)=5.867895e-001; foe(n+1)=9.640775e-001; krok(n+1)=1.000726e-002; ng(n+1)=2.446602e+002;
n=46; farx(n+1)=5.741465e-001; foe(n+1)=9.415460e-001; krok(n+1)=6.227174e-003; ng(n+1)=1.830329e+002;
n=47; farx(n+1)=5.411063e-001; foe(n+1)=8.874035e-001; krok(n+1)=7.931747e-002; ng(n+1)=7.703028e+001;
n=48; farx(n+1)=4.917469e-001; foe(n+1)=8.064917e-001; krok(n+1)=8.892157e-002; ng(n+1)=6.341423e+001;
n=49; farx(n+1)=4.557163e-001; foe(n+1)=7.761047e-001; krok(n+1)=4.663645e-002; ng(n+1)=1.175328e+002;
n=50; farx(n+1)=4.250649e-001; foe(n+1)=7.400754e-001; krok(n+1)=2.896961e-002; ng(n+1)=1.873817e+002;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
