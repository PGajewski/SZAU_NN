%uczenie predyktora oe
clear all;
n=0; farx(n+1)=1.911402e+002; foe(n+1)=1.830131e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.722223e+002; foe(n+1)=1.654011e+002; krok(n+1)=6.696507e-004; ng(n+1)=5.851685e+002;
n=2; farx(n+1)=7.047264e+001; foe(n+1)=6.748097e+001; krok(n+1)=5.647473e-003; ng(n+1)=4.392746e+002;
n=3; farx(n+1)=6.152194e+001; foe(n+1)=5.737464e+001; krok(n+1)=2.156451e-003; ng(n+1)=3.914369e+002;
n=4; farx(n+1)=5.951701e+001; foe(n+1)=5.604513e+001; krok(n+1)=3.065383e-003; ng(n+1)=1.648467e+002;
n=5; farx(n+1)=3.517602e+001; foe(n+1)=4.943318e+001; krok(n+1)=3.045281e-002; ng(n+1)=7.774711e+001;
n=6; farx(n+1)=9.109717e+000; foe(n+1)=3.123867e+001; krok(n+1)=1.196445e-002; ng(n+1)=4.603605e+002;
n=7; farx(n+1)=8.584639e+000; foe(n+1)=3.069511e+001; krok(n+1)=2.885108e-005; ng(n+1)=1.649780e+003;
n=8; farx(n+1)=8.795877e+000; foe(n+1)=2.920136e+001; krok(n+1)=5.200687e-003; ng(n+1)=1.875459e+003;
n=9; farx(n+1)=9.240305e+000; foe(n+1)=2.770207e+001; krok(n+1)=1.072587e-003; ng(n+1)=2.191163e+003;
n=10; farx(n+1)=9.982774e+000; foe(n+1)=2.667694e+001; krok(n+1)=4.042588e-003; ng(n+1)=1.907192e+003;
n=11; farx(n+1)=1.118001e+001; foe(n+1)=2.317244e+001; krok(n+1)=1.840296e-002; ng(n+1)=1.608724e+003;
n=12; farx(n+1)=1.179449e+001; foe(n+1)=2.257977e+001; krok(n+1)=1.857915e-003; ng(n+1)=9.054604e+002;
n=13; farx(n+1)=1.302838e+001; foe(n+1)=2.129523e+001; krok(n+1)=1.835014e-002; ng(n+1)=5.517588e+002;
n=14; farx(n+1)=1.198793e+001; foe(n+1)=1.825737e+001; krok(n+1)=1.612318e-002; ng(n+1)=4.428802e+002;
n=15; farx(n+1)=1.138160e+001; foe(n+1)=1.771106e+001; krok(n+1)=3.949587e-003; ng(n+1)=2.808595e+002;
n=16; farx(n+1)=1.072722e+001; foe(n+1)=1.697713e+001; krok(n+1)=3.291607e-003; ng(n+1)=2.698269e+002;
n=17; farx(n+1)=9.413877e+000; foe(n+1)=1.523662e+001; krok(n+1)=4.969475e-003; ng(n+1)=2.499080e+002;
n=18; farx(n+1)=8.584791e+000; foe(n+1)=1.405050e+001; krok(n+1)=1.251065e-002; ng(n+1)=4.675620e+002;
n=19; farx(n+1)=7.986329e+000; foe(n+1)=1.330816e+001; krok(n+1)=4.952046e-003; ng(n+1)=4.666204e+002;
n=20; farx(n+1)=6.845697e+000; foe(n+1)=1.202693e+001; krok(n+1)=5.946996e-003; ng(n+1)=4.174054e+002;
n=21; farx(n+1)=4.150419e+000; foe(n+1)=8.601959e+000; krok(n+1)=6.802533e-003; ng(n+1)=6.202869e+002;
n=22; farx(n+1)=3.558870e+000; foe(n+1)=7.677281e+000; krok(n+1)=1.944813e-003; ng(n+1)=8.078977e+002;
n=23; farx(n+1)=2.834429e+000; foe(n+1)=6.149202e+000; krok(n+1)=6.718810e-003; ng(n+1)=5.402047e+002;
n=24; farx(n+1)=2.522878e+000; foe(n+1)=5.438319e+000; krok(n+1)=1.703305e-002; ng(n+1)=9.045177e+002;
n=25; farx(n+1)=2.277337e+000; foe(n+1)=5.089964e+000; krok(n+1)=3.011555e-003; ng(n+1)=3.176517e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=2.262004e+000; foe(n+1)=5.041752e+000; krok(n+1)=1.331856e-005; ng(n+1)=2.625346e+002;
n=27; farx(n+1)=2.260593e+000; foe(n+1)=4.902605e+000; krok(n+1)=8.808370e-005; ng(n+1)=1.811953e+002;
n=28; farx(n+1)=2.083902e+000; foe(n+1)=4.153660e+000; krok(n+1)=2.015675e-004; ng(n+1)=2.856029e+002;
n=29; farx(n+1)=1.915881e+000; foe(n+1)=3.629381e+000; krok(n+1)=7.625735e-004; ng(n+1)=2.053446e+002;
n=30; farx(n+1)=1.619930e+000; foe(n+1)=2.805810e+000; krok(n+1)=5.060913e-003; ng(n+1)=1.991379e+002;
n=31; farx(n+1)=1.476519e+000; foe(n+1)=2.435746e+000; krok(n+1)=4.481978e-003; ng(n+1)=6.304775e+002;
n=32; farx(n+1)=9.349223e-001; foe(n+1)=1.940398e+000; krok(n+1)=2.788178e-002; ng(n+1)=2.188079e+002;
n=33; farx(n+1)=7.922368e-001; foe(n+1)=1.782850e+000; krok(n+1)=1.046652e-002; ng(n+1)=1.347423e+002;
n=34; farx(n+1)=7.033336e-001; foe(n+1)=1.682983e+000; krok(n+1)=5.060913e-003; ng(n+1)=3.057051e+002;
n=35; farx(n+1)=6.810836e-001; foe(n+1)=1.501316e+000; krok(n+1)=1.071180e-002; ng(n+1)=1.426447e+002;
n=36; farx(n+1)=6.992877e-001; foe(n+1)=1.397920e+000; krok(n+1)=4.002465e-003; ng(n+1)=2.950417e+002;
n=37; farx(n+1)=6.845901e-001; foe(n+1)=1.304790e+000; krok(n+1)=1.256724e-002; ng(n+1)=3.219670e+002;
n=38; farx(n+1)=6.627067e-001; foe(n+1)=1.219284e+000; krok(n+1)=4.331299e-002; ng(n+1)=3.283470e+001;
n=39; farx(n+1)=6.185499e-001; foe(n+1)=1.169991e+000; krok(n+1)=1.750226e-002; ng(n+1)=1.253029e+002;
n=40; farx(n+1)=6.019344e-001; foe(n+1)=1.143148e+000; krok(n+1)=1.012183e-002; ng(n+1)=1.146012e+002;
n=41; farx(n+1)=5.953332e-001; foe(n+1)=1.099031e+000; krok(n+1)=3.094863e-002; ng(n+1)=4.882265e+001;
n=42; farx(n+1)=5.501313e-001; foe(n+1)=1.021809e+000; krok(n+1)=5.335388e-002; ng(n+1)=4.531083e+001;
n=43; farx(n+1)=5.526028e-001; foe(n+1)=1.001786e+000; krok(n+1)=3.497807e-002; ng(n+1)=5.310664e+001;
n=44; farx(n+1)=5.314669e-001; foe(n+1)=9.665351e-001; krok(n+1)=3.873324e-002; ng(n+1)=7.278501e+001;
n=45; farx(n+1)=5.305122e-001; foe(n+1)=9.431756e-001; krok(n+1)=3.650930e-002; ng(n+1)=6.639157e+001;
n=46; farx(n+1)=4.755416e-001; foe(n+1)=8.764442e-001; krok(n+1)=1.217528e-001; ng(n+1)=1.048667e+002;
n=47; farx(n+1)=4.414724e-001; foe(n+1)=8.422745e-001; krok(n+1)=1.283701e-001; ng(n+1)=1.381292e+002;
n=48; farx(n+1)=4.275906e-001; foe(n+1)=8.251696e-001; krok(n+1)=4.438543e-002; ng(n+1)=1.416144e+002;
n=49; farx(n+1)=4.201275e-001; foe(n+1)=8.161311e-001; krok(n+1)=8.005809e-002; ng(n+1)=2.793743e+001;
n=50; farx(n+1)=4.066827e-001; foe(n+1)=7.884966e-001; krok(n+1)=2.863617e-001; ng(n+1)=2.480908e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)