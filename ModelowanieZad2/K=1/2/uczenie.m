%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.242349e+002; foe(n+1)=2.242292e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.777381e+002; foe(n+1)=1.777218e+002; krok(n+1)=5.005982e-004; ng(n+1)=3.472048e+002;
n=2; farx(n+1)=6.318551e+001; foe(n+1)=6.317147e+001; krok(n+1)=3.771383e-002; ng(n+1)=3.773356e+001;
n=3; farx(n+1)=6.254116e+001; foe(n+1)=6.244260e+001; krok(n+1)=1.582138e-004; ng(n+1)=2.013676e+002;
n=4; farx(n+1)=7.276960e+001; foe(n+1)=5.984609e+001; krok(n+1)=2.093303e-002; ng(n+1)=2.203341e+002;
n=5; farx(n+1)=7.235319e+001; foe(n+1)=5.870918e+001; krok(n+1)=1.394089e-002; ng(n+1)=7.681721e+001;
n=6; farx(n+1)=8.231890e+001; foe(n+1)=5.537817e+001; krok(n+1)=1.655202e-001; ng(n+1)=9.829999e+001;
n=7; farx(n+1)=3.442049e+001; foe(n+1)=4.551738e+001; krok(n+1)=1.487414e-001; ng(n+1)=5.955010e+001;
n=8; farx(n+1)=2.846755e+001; foe(n+1)=4.459147e+001; krok(n+1)=4.126056e-003; ng(n+1)=1.593332e+002;
n=9; farx(n+1)=2.094141e+001; foe(n+1)=4.323079e+001; krok(n+1)=1.725161e-002; ng(n+1)=2.388128e+002;
n=10; farx(n+1)=8.110465e+000; foe(n+1)=3.457292e+001; krok(n+1)=3.288994e-001; ng(n+1)=4.064994e+002;
n=11; farx(n+1)=3.285098e+000; foe(n+1)=1.370828e+001; krok(n+1)=2.577747e-001; ng(n+1)=8.866885e+002;
n=12; farx(n+1)=2.673503e+000; foe(n+1)=1.228560e+001; krok(n+1)=8.973423e-002; ng(n+1)=2.984514e+002;
n=13; farx(n+1)=2.387782e+000; foe(n+1)=1.156067e+001; krok(n+1)=1.381465e-001; ng(n+1)=1.870760e+002;
n=14; farx(n+1)=2.250136e+000; foe(n+1)=1.110723e+001; krok(n+1)=2.725288e-001; ng(n+1)=8.719226e+001;
n=15; farx(n+1)=1.897865e+000; foe(n+1)=1.000925e+001; krok(n+1)=2.560234e+000; ng(n+1)=1.786290e+002;
n=16; farx(n+1)=1.030584e+000; foe(n+1)=8.490143e+000; krok(n+1)=1.817806e+000; ng(n+1)=1.052631e+002;
n=17; farx(n+1)=1.005542e+000; foe(n+1)=8.365563e+000; krok(n+1)=5.461800e-001; ng(n+1)=9.276127e+001;
n=18; farx(n+1)=1.005422e+000; foe(n+1)=8.267686e+000; krok(n+1)=1.498210e+000; ng(n+1)=3.400615e+001;
n=19; farx(n+1)=9.852711e-001; foe(n+1)=8.249458e+000; krok(n+1)=5.792353e-001; ng(n+1)=2.816129e+001;
n=20; farx(n+1)=9.687668e-001; foe(n+1)=8.246025e+000; krok(n+1)=1.420334e+000; ng(n+1)=1.054268e+001;
n=21; farx(n+1)=9.602231e-001; foe(n+1)=8.235913e+000; krok(n+1)=9.540617e+000; ng(n+1)=1.216587e+001;
n=22; farx(n+1)=9.834264e-001; foe(n+1)=8.207833e+000; krok(n+1)=2.208206e+000; ng(n+1)=1.495874e+001;
n=23; farx(n+1)=9.898063e-001; foe(n+1)=8.200349e+000; krok(n+1)=1.322970e+000; ng(n+1)=2.245316e+001;
n=24; farx(n+1)=1.004247e+000; foe(n+1)=8.198233e+000; krok(n+1)=1.157348e+000; ng(n+1)=7.595619e+000;
n=25; farx(n+1)=1.003969e+000; foe(n+1)=8.197953e+000; krok(n+1)=8.922169e-001; ng(n+1)=1.939388e+000;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.003800e+000; foe(n+1)=8.197935e+000; krok(n+1)=4.037383e-005; ng(n+1)=1.678865e+000;
n=27; farx(n+1)=1.003728e+000; foe(n+1)=8.197909e+000; krok(n+1)=7.745847e-005; ng(n+1)=1.197740e+000;
n=28; farx(n+1)=1.003389e+000; foe(n+1)=8.197905e+000; krok(n+1)=1.137613e-003; ng(n+1)=1.596131e-001;
n=29; farx(n+1)=1.002426e+000; foe(n+1)=8.197883e+000; krok(n+1)=1.767909e-002; ng(n+1)=8.456831e-002;
n=30; farx(n+1)=1.003228e+000; foe(n+1)=8.197209e+000; krok(n+1)=4.268310e-001; ng(n+1)=9.151137e-002;
n=31; farx(n+1)=1.032204e+000; foe(n+1)=8.195184e+000; krok(n+1)=7.744164e-001; ng(n+1)=9.134610e-002;
n=32; farx(n+1)=1.040214e+000; foe(n+1)=8.194462e+000; krok(n+1)=8.470464e-002; ng(n+1)=1.995732e-001;
n=33; farx(n+1)=1.047488e+000; foe(n+1)=8.194084e+000; krok(n+1)=1.663330e-001; ng(n+1)=4.703706e-001;
n=34; farx(n+1)=1.043974e+000; foe(n+1)=8.193873e+000; krok(n+1)=3.390079e-001; ng(n+1)=7.598252e-001;
n=35; farx(n+1)=1.045535e+000; foe(n+1)=8.193853e+000; krok(n+1)=9.634903e-001; ng(n+1)=6.812525e-001;
n=36; farx(n+1)=1.045477e+000; foe(n+1)=8.193852e+000; krok(n+1)=9.282331e-001; ng(n+1)=8.500061e-002;
n=37; farx(n+1)=1.045510e+000; foe(n+1)=8.193852e+000; krok(n+1)=1.104103e+000; ng(n+1)=2.289461e-003;
n=38; farx(n+1)=1.045510e+000; foe(n+1)=8.193852e+000; krok(n+1)=2.788618e-006; ng(n+1)=2.704077e-004;
n=39; farx(n+1)=1.045510e+000; foe(n+1)=8.193852e+000; krok(n+1)=5.376572e-006; ng(n+1)=2.704068e-004;
n=40; farx(n+1)=1.045510e+000; foe(n+1)=8.193852e+000; krok(n+1)=1.017823e-008; ng(n+1)=2.704054e-004;
n=41; farx(n+1)=1.045510e+000; foe(n+1)=8.193852e+000; krok(n+1)=2.473884e-010; ng(n+1)=2.704054e-004;
n=42; farx(n+1)=1.045510e+000; foe(n+1)=8.193852e+000; krok(n+1)=2.344936e-009; ng(n+1)=2.704054e-004;
 % z�y kierunek w metodzie zm - odnowa 
n=43; farx(n+1)=1.045510e+000; foe(n+1)=8.193852e+000; krok(n+1)=5.691308e-005; ng(n+1)=2.704054e-004;
n=44; farx(n+1)=1.045510e+000; foe(n+1)=8.193852e+000; krok(n+1)=7.687793e-005; ng(n+1)=2.529682e-004;
n=45; farx(n+1)=1.045510e+000; foe(n+1)=8.193852e+000; krok(n+1)=2.139878e-005; ng(n+1)=7.403689e-005;
n=46; farx(n+1)=1.045510e+000; foe(n+1)=8.193852e+000; krok(n+1)=3.090981e-005; ng(n+1)=7.325159e-005;
n=47; farx(n+1)=1.045510e+000; foe(n+1)=8.193852e+000; krok(n+1)=1.562594e-007; ng(n+1)=7.325266e-005;
n=48; farx(n+1)=1.045510e+000; foe(n+1)=8.193852e+000; krok(n+1)=1.720987e-008; ng(n+1)=7.325254e-005;
n=49; farx(n+1)=1.045510e+000; foe(n+1)=8.193852e+000; krok(n+1)=3.658996e-010; ng(n+1)=7.325254e-005;
n=50; farx(n+1)=1.045509e+000; foe(n+1)=8.193852e+000; krok(n+1)=2.152030e-004; ng(n+1)=7.325254e-005;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)