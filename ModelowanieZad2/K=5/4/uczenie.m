%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.843908e+002; foe(n+1)=2.862665e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.763277e+002; foe(n+1)=1.777972e+002; krok(n+1)=4.912482e-004; ng(n+1)=1.000912e+003;
n=2; farx(n+1)=5.515815e+001; foe(n+1)=5.538305e+001; krok(n+1)=2.128042e-002; ng(n+1)=3.521776e+002;
n=3; farx(n+1)=5.117459e+001; foe(n+1)=5.405084e+001; krok(n+1)=4.312903e-003; ng(n+1)=9.343209e+001;
n=4; farx(n+1)=3.576459e+001; foe(n+1)=4.823335e+001; krok(n+1)=1.970836e-003; ng(n+1)=2.967203e+002;
n=5; farx(n+1)=1.500790e+001; foe(n+1)=2.842920e+001; krok(n+1)=1.310706e-003; ng(n+1)=4.459791e+002;
n=6; farx(n+1)=1.456916e+001; foe(n+1)=2.672389e+001; krok(n+1)=4.430569e-004; ng(n+1)=7.288218e+002;
n=7; farx(n+1)=1.398541e+001; foe(n+1)=2.613066e+001; krok(n+1)=5.711090e-003; ng(n+1)=4.654154e+002;
n=8; farx(n+1)=1.430635e+001; foe(n+1)=2.347746e+001; krok(n+1)=1.078150e-002; ng(n+1)=3.272384e+002;
n=9; farx(n+1)=1.358366e+001; foe(n+1)=2.259338e+001; krok(n+1)=5.478088e-003; ng(n+1)=1.123837e+002;
n=10; farx(n+1)=1.201943e+001; foe(n+1)=2.054815e+001; krok(n+1)=8.704305e-003; ng(n+1)=2.330835e+002;
n=11; farx(n+1)=1.118803e+001; foe(n+1)=1.974287e+001; krok(n+1)=4.974973e-003; ng(n+1)=1.640351e+002;
n=12; farx(n+1)=9.184579e+000; foe(n+1)=1.777039e+001; krok(n+1)=7.530795e-003; ng(n+1)=2.728369e+002;
n=13; farx(n+1)=7.007381e+000; foe(n+1)=1.621565e+001; krok(n+1)=1.254246e-002; ng(n+1)=4.391673e+002;
n=14; farx(n+1)=3.569845e+000; foe(n+1)=1.301899e+001; krok(n+1)=1.574016e-002; ng(n+1)=4.737418e+002;
n=15; farx(n+1)=1.994132e+000; foe(n+1)=9.196836e+000; krok(n+1)=2.815703e-003; ng(n+1)=1.444949e+003;
n=16; farx(n+1)=1.633712e+000; foe(n+1)=7.773541e+000; krok(n+1)=3.334617e-003; ng(n+1)=7.989457e+002;
n=17; farx(n+1)=1.629415e+000; foe(n+1)=7.572184e+000; krok(n+1)=3.778589e-003; ng(n+1)=1.657758e+002;
n=18; farx(n+1)=1.654673e+000; foe(n+1)=6.535953e+000; krok(n+1)=9.436540e-003; ng(n+1)=2.353554e+002;
n=19; farx(n+1)=1.623764e+000; foe(n+1)=6.243497e+000; krok(n+1)=1.809047e-002; ng(n+1)=2.478593e+002;
n=20; farx(n+1)=1.463650e+000; foe(n+1)=5.144334e+000; krok(n+1)=3.403861e-002; ng(n+1)=2.319944e+002;
n=21; farx(n+1)=1.258987e+000; foe(n+1)=4.050390e+000; krok(n+1)=1.304848e-001; ng(n+1)=1.866006e+002;
n=22; farx(n+1)=1.184726e+000; foe(n+1)=3.426037e+000; krok(n+1)=1.775417e-001; ng(n+1)=2.024881e+002;
n=23; farx(n+1)=8.451712e-001; foe(n+1)=2.213245e+000; krok(n+1)=2.488881e-001; ng(n+1)=3.082120e+002;
n=24; farx(n+1)=6.470754e-001; foe(n+1)=1.480309e+000; krok(n+1)=3.435737e-001; ng(n+1)=1.925345e+002;
n=25; farx(n+1)=4.925585e-001; foe(n+1)=1.235545e+000; krok(n+1)=1.372911e-001; ng(n+1)=1.541918e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=4.893603e-001; foe(n+1)=1.172718e+000; krok(n+1)=2.253084e-005; ng(n+1)=1.478464e+002;
n=27; farx(n+1)=4.754740e-001; foe(n+1)=1.066769e+000; krok(n+1)=5.665485e-005; ng(n+1)=1.395222e+002;
n=28; farx(n+1)=4.792913e-001; foe(n+1)=1.036643e+000; krok(n+1)=5.926151e-005; ng(n+1)=6.358975e+001;
n=29; farx(n+1)=4.544381e-001; foe(n+1)=9.878917e-001; krok(n+1)=8.108296e-003; ng(n+1)=1.459296e+001;
n=30; farx(n+1)=4.255727e-001; foe(n+1)=9.087441e-001; krok(n+1)=8.128862e-003; ng(n+1)=1.770414e+001;
n=31; farx(n+1)=4.151129e-001; foe(n+1)=8.853608e-001; krok(n+1)=1.725161e-002; ng(n+1)=3.757901e+001;
n=32; farx(n+1)=3.776543e-001; foe(n+1)=7.981957e-001; krok(n+1)=1.438335e-002; ng(n+1)=6.637873e+001;
n=33; farx(n+1)=3.670046e-001; foe(n+1)=7.629897e-001; krok(n+1)=1.056599e-002; ng(n+1)=1.240952e+002;
n=34; farx(n+1)=3.574166e-001; foe(n+1)=7.377332e-001; krok(n+1)=5.335388e-002; ng(n+1)=6.146041e+001;
n=35; farx(n+1)=3.573774e-001; foe(n+1)=6.519047e-001; krok(n+1)=1.556123e-001; ng(n+1)=2.223138e+001;
n=36; farx(n+1)=3.632389e-001; foe(n+1)=6.315728e-001; krok(n+1)=1.015974e-001; ng(n+1)=3.912586e+001;
n=37; farx(n+1)=3.642141e-001; foe(n+1)=6.094679e-001; krok(n+1)=1.687792e-001; ng(n+1)=5.047180e+001;
n=38; farx(n+1)=3.533592e-001; foe(n+1)=5.822048e-001; krok(n+1)=6.306675e-002; ng(n+1)=5.973173e+001;
n=39; farx(n+1)=3.417487e-001; foe(n+1)=5.696515e-001; krok(n+1)=2.498965e-001; ng(n+1)=2.355708e+001;
n=40; farx(n+1)=3.537485e-001; foe(n+1)=5.556350e-001; krok(n+1)=2.271963e-001; ng(n+1)=3.320265e+001;
n=41; farx(n+1)=3.639313e-001; foe(n+1)=5.436256e-001; krok(n+1)=2.953927e-001; ng(n+1)=4.032751e+001;
n=42; farx(n+1)=3.706405e-001; foe(n+1)=5.226664e-001; krok(n+1)=1.298711e-001; ng(n+1)=6.987995e+001;
n=43; farx(n+1)=3.748653e-001; foe(n+1)=5.048602e-001; krok(n+1)=2.990259e-001; ng(n+1)=2.812844e+001;
n=44; farx(n+1)=3.742957e-001; foe(n+1)=4.774746e-001; krok(n+1)=2.833785e-001; ng(n+1)=5.736833e+001;
n=45; farx(n+1)=3.439455e-001; foe(n+1)=4.629606e-001; krok(n+1)=4.741921e-001; ng(n+1)=1.390023e+001;
n=46; farx(n+1)=3.410877e-001; foe(n+1)=4.495702e-001; krok(n+1)=4.268310e-001; ng(n+1)=3.606149e+001;
n=47; farx(n+1)=3.448025e-001; foe(n+1)=4.426651e-001; krok(n+1)=1.666377e-001; ng(n+1)=3.089954e+001;
n=48; farx(n+1)=3.556284e-001; foe(n+1)=4.339732e-001; krok(n+1)=4.811170e-001; ng(n+1)=2.556513e+001;
n=49; farx(n+1)=3.505207e-001; foe(n+1)=4.212807e-001; krok(n+1)=5.014424e-001; ng(n+1)=2.065868e+001;
n=50; farx(n+1)=3.397123e-001; foe(n+1)=4.110982e-001; krok(n+1)=4.228884e-001; ng(n+1)=3.083884e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
