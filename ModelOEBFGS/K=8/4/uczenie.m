%uczenie predyktora oe
clear all;
n=0; farx(n+1)=1.775710e+002; foe(n+1)=1.827717e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.558581e+002; foe(n+1)=1.621904e+002; krok(n+1)=5.750926e-004; ng(n+1)=7.259774e+002;
n=2; farx(n+1)=6.179140e+001; foe(n+1)=6.952278e+001; krok(n+1)=5.320104e-003; ng(n+1)=4.899378e+002;
n=3; farx(n+1)=5.594643e+001; foe(n+1)=6.112104e+001; krok(n+1)=2.823736e-003; ng(n+1)=4.119786e+002;
n=4; farx(n+1)=5.324211e+001; foe(n+1)=6.044106e+001; krok(n+1)=6.118871e-003; ng(n+1)=6.907156e+001;
n=5; farx(n+1)=2.531153e+001; foe(n+1)=5.339747e+001; krok(n+1)=2.439969e-002; ng(n+1)=1.039452e+002;
n=6; farx(n+1)=1.034384e+001; foe(n+1)=4.642670e+001; krok(n+1)=5.139053e-003; ng(n+1)=6.223092e+002;
n=7; farx(n+1)=7.607637e+000; foe(n+1)=4.527423e+001; krok(n+1)=1.528930e-004; ng(n+1)=1.414456e+003;
n=8; farx(n+1)=6.185605e+000; foe(n+1)=4.381829e+001; krok(n+1)=7.558357e-003; ng(n+1)=2.080147e+003;
n=9; farx(n+1)=5.292347e+000; foe(n+1)=4.210298e+001; krok(n+1)=7.138862e-004; ng(n+1)=2.497246e+003;
n=10; farx(n+1)=4.680687e+000; foe(n+1)=4.081855e+001; krok(n+1)=1.561071e-003; ng(n+1)=3.479818e+003;
n=11; farx(n+1)=2.681826e+000; foe(n+1)=1.790594e+001; krok(n+1)=1.865725e-002; ng(n+1)=4.532458e+003;
n=12; farx(n+1)=2.575297e+000; foe(n+1)=1.761184e+001; krok(n+1)=7.980756e-005; ng(n+1)=1.484314e+003;
n=13; farx(n+1)=2.635797e+000; foe(n+1)=1.276196e+001; krok(n+1)=4.525276e-003; ng(n+1)=1.707125e+003;
n=14; farx(n+1)=2.530302e+000; foe(n+1)=1.230968e+001; krok(n+1)=6.254538e-004; ng(n+1)=6.751532e+002;
n=15; farx(n+1)=2.315232e+000; foe(n+1)=1.129638e+001; krok(n+1)=8.580693e-003; ng(n+1)=7.662446e+002;
n=16; farx(n+1)=1.346563e+000; foe(n+1)=8.187356e+000; krok(n+1)=4.799811e-002; ng(n+1)=2.120344e+002;
n=17; farx(n+1)=1.194126e+000; foe(n+1)=7.800069e+000; krok(n+1)=3.584297e-003; ng(n+1)=1.631875e+002;
n=18; farx(n+1)=1.111521e+000; foe(n+1)=7.446506e+000; krok(n+1)=4.325549e-004; ng(n+1)=5.223797e+002;
n=19; farx(n+1)=1.001743e+000; foe(n+1)=7.213599e+000; krok(n+1)=3.258189e-003; ng(n+1)=4.681148e+002;
n=20; farx(n+1)=9.032103e-001; foe(n+1)=6.094650e+000; krok(n+1)=1.189399e-002; ng(n+1)=4.022647e+002;
n=21; farx(n+1)=9.373309e-001; foe(n+1)=5.664741e+000; krok(n+1)=5.637357e-003; ng(n+1)=3.963792e+002;
n=22; farx(n+1)=9.182133e-001; foe(n+1)=4.952944e+000; krok(n+1)=1.776017e-002; ng(n+1)=4.989147e+002;
n=23; farx(n+1)=8.882761e-001; foe(n+1)=4.375190e+000; krok(n+1)=4.815473e-003; ng(n+1)=5.497802e+002;
n=24; farx(n+1)=8.322730e-001; foe(n+1)=4.090127e+000; krok(n+1)=1.142218e-002; ng(n+1)=4.581940e+002;
n=25; farx(n+1)=7.855273e-001; foe(n+1)=3.750940e+000; krok(n+1)=3.294851e-002; ng(n+1)=3.541773e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=7.717972e-001; foe(n+1)=3.537800e+000; krok(n+1)=1.138543e-005; ng(n+1)=6.063246e+002;
n=27; farx(n+1)=7.562570e-001; foe(n+1)=3.504488e+000; krok(n+1)=1.955288e-004; ng(n+1)=5.382261e+001;
n=28; farx(n+1)=6.522675e-001; foe(n+1)=2.785289e+000; krok(n+1)=8.566255e-004; ng(n+1)=1.174488e+002;
n=29; farx(n+1)=6.107517e-001; foe(n+1)=2.095777e+000; krok(n+1)=2.779644e-004; ng(n+1)=2.731566e+002;
n=30; farx(n+1)=6.207180e-001; foe(n+1)=1.960465e+000; krok(n+1)=7.163682e-003; ng(n+1)=2.184979e+002;
n=31; farx(n+1)=6.260704e-001; foe(n+1)=1.744394e+000; krok(n+1)=1.370601e-002; ng(n+1)=3.647474e+002;
n=32; farx(n+1)=5.944121e-001; foe(n+1)=1.681070e+000; krok(n+1)=8.402186e-003; ng(n+1)=1.439551e+002;
n=33; farx(n+1)=5.448072e-001; foe(n+1)=1.588906e+000; krok(n+1)=1.538463e-002; ng(n+1)=1.029699e+002;
n=34; farx(n+1)=5.128231e-001; foe(n+1)=1.394874e+000; krok(n+1)=5.514490e-002; ng(n+1)=1.833849e+002;
n=35; farx(n+1)=5.069553e-001; foe(n+1)=1.361983e+000; krok(n+1)=3.625910e-003; ng(n+1)=1.258704e+002;
n=36; farx(n+1)=4.878947e-001; foe(n+1)=1.295712e+000; krok(n+1)=2.079162e-002; ng(n+1)=9.018809e+001;
n=37; farx(n+1)=4.776725e-001; foe(n+1)=1.213422e+000; krok(n+1)=3.130011e-002; ng(n+1)=1.261977e+002;
n=38; farx(n+1)=4.816306e-001; foe(n+1)=1.180937e+000; krok(n+1)=1.000726e-002; ng(n+1)=1.093403e+002;
n=39; farx(n+1)=4.872616e-001; foe(n+1)=1.151395e+000; krok(n+1)=1.684199e-002; ng(n+1)=1.075356e+002;
n=40; farx(n+1)=5.001338e-001; foe(n+1)=1.110837e+000; krok(n+1)=4.134280e-002; ng(n+1)=1.792610e+001;
n=41; farx(n+1)=5.038348e-001; foe(n+1)=1.052389e+000; krok(n+1)=8.430727e-002; ng(n+1)=7.798609e+001;
n=42; farx(n+1)=5.036730e-001; foe(n+1)=1.042081e+000; krok(n+1)=2.780135e-002; ng(n+1)=7.796076e+001;
n=43; farx(n+1)=4.937311e-001; foe(n+1)=1.002416e+000; krok(n+1)=1.011873e-001; ng(n+1)=8.320853e+001;
n=44; farx(n+1)=4.687232e-001; foe(n+1)=9.637467e-001; krok(n+1)=8.569436e-002; ng(n+1)=1.198652e+002;
n=45; farx(n+1)=4.550351e-001; foe(n+1)=9.384853e-001; krok(n+1)=8.035594e-002; ng(n+1)=8.576907e+001;
n=46; farx(n+1)=4.522233e-001; foe(n+1)=9.293980e-001; krok(n+1)=6.776891e-003; ng(n+1)=1.137298e+002;
n=47; farx(n+1)=4.427312e-001; foe(n+1)=9.087194e-001; krok(n+1)=1.416892e-001; ng(n+1)=1.784841e+001;
n=48; farx(n+1)=4.449889e-001; foe(n+1)=8.898141e-001; krok(n+1)=3.542231e-002; ng(n+1)=7.955191e+001;
n=49; farx(n+1)=4.470049e-001; foe(n+1)=8.845949e-001; krok(n+1)=4.732423e-002; ng(n+1)=6.124550e+001;
n=50; farx(n+1)=4.447665e-001; foe(n+1)=8.631010e-001; krok(n+1)=6.640781e-002; ng(n+1)=5.192419e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
