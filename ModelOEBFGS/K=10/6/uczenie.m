%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.244286e+002; foe(n+1)=2.249322e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.708185e+002; foe(n+1)=1.721545e+002; krok(n+1)=4.927090e-004; ng(n+1)=1.024150e+003;
n=2; farx(n+1)=5.866520e+001; foe(n+1)=6.303196e+001; krok(n+1)=9.157950e-003; ng(n+1)=4.692395e+002;
n=3; farx(n+1)=5.712659e+001; foe(n+1)=6.138854e+001; krok(n+1)=1.155113e-003; ng(n+1)=2.116460e+002;
n=4; farx(n+1)=5.123906e+001; foe(n+1)=6.056864e+001; krok(n+1)=4.969883e-003; ng(n+1)=6.804375e+001;
n=5; farx(n+1)=4.996096e+000; foe(n+1)=3.224745e+001; krok(n+1)=1.799391e-002; ng(n+1)=1.407944e+002;
n=6; farx(n+1)=4.248380e+000; foe(n+1)=3.161957e+001; krok(n+1)=2.007443e-005; ng(n+1)=2.532033e+003;
n=7; farx(n+1)=4.595143e+000; foe(n+1)=3.012888e+001; krok(n+1)=8.403478e-003; ng(n+1)=3.293589e+003;
n=8; farx(n+1)=5.528867e+000; foe(n+1)=2.359505e+001; krok(n+1)=7.624904e-004; ng(n+1)=3.295745e+003;
n=9; farx(n+1)=6.141737e+000; foe(n+1)=2.122760e+001; krok(n+1)=3.128460e-003; ng(n+1)=1.468985e+003;
n=10; farx(n+1)=5.482112e+000; foe(n+1)=1.960079e+001; krok(n+1)=9.082851e-003; ng(n+1)=7.486625e+002;
n=11; farx(n+1)=3.669289e+000; foe(n+1)=1.537284e+001; krok(n+1)=5.095665e-003; ng(n+1)=6.522473e+002;
n=12; farx(n+1)=2.868909e+000; foe(n+1)=1.338520e+001; krok(n+1)=3.778589e-003; ng(n+1)=1.527983e+003;
n=13; farx(n+1)=2.479002e+000; foe(n+1)=1.221283e+001; krok(n+1)=1.427772e-003; ng(n+1)=1.119630e+003;
n=14; farx(n+1)=2.077812e+000; foe(n+1)=1.115788e+001; krok(n+1)=1.000726e-002; ng(n+1)=6.068674e+002;
n=15; farx(n+1)=1.737741e+000; foe(n+1)=9.546815e+000; krok(n+1)=1.265228e-003; ng(n+1)=1.503682e+003;
n=16; farx(n+1)=1.666208e+000; foe(n+1)=9.131934e+000; krok(n+1)=2.774090e-003; ng(n+1)=5.440271e+002;
n=17; farx(n+1)=1.573551e+000; foe(n+1)=8.557212e+000; krok(n+1)=4.815473e-003; ng(n+1)=1.117040e+003;
n=18; farx(n+1)=1.499890e+000; foe(n+1)=7.647477e+000; krok(n+1)=3.206931e-003; ng(n+1)=1.314010e+003;
n=19; farx(n+1)=1.425589e+000; foe(n+1)=7.174086e+000; krok(n+1)=1.360507e-002; ng(n+1)=1.080693e+003;
n=20; farx(n+1)=1.329627e+000; foe(n+1)=6.187199e+000; krok(n+1)=7.251821e-003; ng(n+1)=1.242653e+003;
n=21; farx(n+1)=1.224477e+000; foe(n+1)=5.694098e+000; krok(n+1)=2.647552e-003; ng(n+1)=8.348884e+002;
n=22; farx(n+1)=1.117919e+000; foe(n+1)=4.973050e+000; krok(n+1)=1.270603e-002; ng(n+1)=7.348811e+002;
n=23; farx(n+1)=1.024290e+000; foe(n+1)=4.486682e+000; krok(n+1)=1.667309e-003; ng(n+1)=5.134346e+002;
n=24; farx(n+1)=9.286094e-001; foe(n+1)=4.050585e+000; krok(n+1)=3.253129e-003; ng(n+1)=5.850082e+002;
n=25; farx(n+1)=8.926660e-001; foe(n+1)=3.810949e+000; krok(n+1)=6.353015e-003; ng(n+1)=5.508694e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=8.904447e-001; foe(n+1)=3.640155e+000; krok(n+1)=6.371971e-006; ng(n+1)=7.090453e+002;
n=27; farx(n+1)=8.878702e-001; foe(n+1)=3.620875e+000; krok(n+1)=1.342816e-005; ng(n+1)=1.957527e+002;
n=28; farx(n+1)=8.969096e-001; foe(n+1)=3.548381e+000; krok(n+1)=2.497184e-004; ng(n+1)=1.003126e+002;
n=29; farx(n+1)=8.243516e-001; foe(n+1)=3.159678e+000; krok(n+1)=4.150488e-003; ng(n+1)=6.388268e+001;
n=30; farx(n+1)=6.334753e-001; foe(n+1)=2.172192e+000; krok(n+1)=1.208515e-002; ng(n+1)=1.044504e+002;
n=31; farx(n+1)=6.172537e-001; foe(n+1)=2.064470e+000; krok(n+1)=1.812955e-003; ng(n+1)=6.713695e+002;
n=32; farx(n+1)=5.976313e-001; foe(n+1)=1.960758e+000; krok(n+1)=3.704626e-003; ng(n+1)=2.393752e+002;
n=33; farx(n+1)=5.858825e-001; foe(n+1)=1.771083e+000; krok(n+1)=1.809047e-002; ng(n+1)=1.227692e+002;
n=34; farx(n+1)=5.906983e-001; foe(n+1)=1.693610e+000; krok(n+1)=9.625525e-003; ng(n+1)=2.890576e+002;
n=35; farx(n+1)=6.246731e-001; foe(n+1)=1.543960e+000; krok(n+1)=2.366103e-002; ng(n+1)=1.183342e+002;
n=36; farx(n+1)=6.411447e-001; foe(n+1)=1.442279e+000; krok(n+1)=8.202462e-003; ng(n+1)=2.623486e+002;
n=37; farx(n+1)=6.487118e-001; foe(n+1)=1.388808e+000; krok(n+1)=8.806488e-003; ng(n+1)=2.339070e+002;
n=38; farx(n+1)=6.285127e-001; foe(n+1)=1.272521e+000; krok(n+1)=3.244437e-002; ng(n+1)=1.180729e+002;
n=39; farx(n+1)=6.111748e-001; foe(n+1)=1.210886e+000; krok(n+1)=8.023134e-003; ng(n+1)=2.742811e+002;
n=40; farx(n+1)=6.079871e-001; foe(n+1)=1.191209e+000; krok(n+1)=7.242402e-003; ng(n+1)=1.127732e+002;
n=41; farx(n+1)=5.889331e-001; foe(n+1)=1.128154e+000; krok(n+1)=1.008497e-001; ng(n+1)=1.335586e+002;
n=42; farx(n+1)=5.642282e-001; foe(n+1)=1.080148e+000; krok(n+1)=1.131031e-002; ng(n+1)=3.082274e+002;
n=43; farx(n+1)=5.436927e-001; foe(n+1)=1.030468e+000; krok(n+1)=1.761298e-002; ng(n+1)=1.510815e+002;
n=44; farx(n+1)=4.807902e-001; foe(n+1)=9.082768e-001; krok(n+1)=6.385486e-002; ng(n+1)=8.878046e+001;
n=45; farx(n+1)=4.536731e-001; foe(n+1)=8.522254e-001; krok(n+1)=3.065383e-003; ng(n+1)=3.122492e+002;
n=46; farx(n+1)=4.339853e-001; foe(n+1)=8.268311e-001; krok(n+1)=4.727372e-003; ng(n+1)=1.217870e+002;
n=47; farx(n+1)=3.953057e-001; foe(n+1)=7.999390e-001; krok(n+1)=4.838350e-002; ng(n+1)=1.297564e+002;
n=48; farx(n+1)=3.771423e-001; foe(n+1)=7.823840e-001; krok(n+1)=1.703305e-002; ng(n+1)=1.248341e+002;
n=49; farx(n+1)=3.532670e-001; foe(n+1)=7.490550e-001; krok(n+1)=6.863735e-002; ng(n+1)=9.726973e+001;
n=50; farx(n+1)=3.286607e-001; foe(n+1)=6.681110e-001; krok(n+1)=1.057221e-001; ng(n+1)=1.201653e+002;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)