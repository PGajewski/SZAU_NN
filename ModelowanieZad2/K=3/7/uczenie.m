%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.732793e+002; foe(n+1)=2.665933e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.801091e+002; foe(n+1)=1.744183e+002; krok(n+1)=5.113081e-004; ng(n+1)=8.089633e+002;
n=2; farx(n+1)=6.295110e+001; foe(n+1)=5.876171e+001; krok(n+1)=1.370811e-002; ng(n+1)=2.545379e+002;
n=3; farx(n+1)=5.981543e+001; foe(n+1)=5.562255e+001; krok(n+1)=3.595838e-003; ng(n+1)=1.562070e+002;
n=4; farx(n+1)=5.762375e+001; foe(n+1)=5.414864e+001; krok(n+1)=1.882699e-003; ng(n+1)=1.124514e+002;
n=5; farx(n+1)=4.038254e+001; foe(n+1)=4.723665e+001; krok(n+1)=4.002904e-002; ng(n+1)=6.205576e+001;
n=6; farx(n+1)=1.561142e+001; foe(n+1)=3.086309e+001; krok(n+1)=4.569519e-002; ng(n+1)=1.305518e+002;
n=7; farx(n+1)=1.352152e+001; foe(n+1)=2.881348e+001; krok(n+1)=6.180774e-004; ng(n+1)=4.096330e+002;
n=8; farx(n+1)=1.355622e+001; foe(n+1)=2.629383e+001; krok(n+1)=7.314625e-003; ng(n+1)=6.091449e+002;
n=9; farx(n+1)=1.284828e+001; foe(n+1)=2.408496e+001; krok(n+1)=2.598758e-002; ng(n+1)=4.432791e+002;
n=10; farx(n+1)=1.232701e+001; foe(n+1)=2.246242e+001; krok(n+1)=1.274938e-002; ng(n+1)=1.691066e+002;
n=11; farx(n+1)=1.120470e+001; foe(n+1)=2.117849e+001; krok(n+1)=1.053841e-002; ng(n+1)=3.591918e+002;
n=12; farx(n+1)=1.015138e+001; foe(n+1)=2.047559e+001; krok(n+1)=5.767896e-003; ng(n+1)=3.666807e+002;
n=13; farx(n+1)=1.221275e+001; foe(n+1)=1.944798e+001; krok(n+1)=4.517978e-002; ng(n+1)=5.658050e+002;
n=14; farx(n+1)=1.227862e+001; foe(n+1)=1.785031e+001; krok(n+1)=3.023343e-002; ng(n+1)=5.140639e+002;
n=15; farx(n+1)=1.183224e+001; foe(n+1)=1.761519e+001; krok(n+1)=1.023953e-002; ng(n+1)=1.910790e+002;
n=16; farx(n+1)=9.625879e+000; foe(n+1)=1.628650e+001; krok(n+1)=2.476948e-002; ng(n+1)=1.993078e+002;
n=17; farx(n+1)=4.695481e+000; foe(n+1)=1.321776e+001; krok(n+1)=2.975933e-001; ng(n+1)=1.267192e+002;
n=18; farx(n+1)=2.181752e+000; foe(n+1)=1.195079e+001; krok(n+1)=4.015963e-001; ng(n+1)=1.510025e+002;
n=19; farx(n+1)=1.765008e+000; foe(n+1)=9.504612e+000; krok(n+1)=3.262230e-001; ng(n+1)=3.767131e+002;
n=20; farx(n+1)=1.595334e+000; foe(n+1)=8.439420e+000; krok(n+1)=4.114269e-001; ng(n+1)=2.604203e+002;
n=21; farx(n+1)=1.544499e+000; foe(n+1)=7.999453e+000; krok(n+1)=8.973423e-002; ng(n+1)=1.685414e+002;
n=22; farx(n+1)=1.506739e+000; foe(n+1)=7.621370e+000; krok(n+1)=4.842911e-002; ng(n+1)=2.597883e+002;
n=23; farx(n+1)=1.277228e+000; foe(n+1)=6.602099e+000; krok(n+1)=1.120145e+000; ng(n+1)=3.026518e+002;
n=24; farx(n+1)=1.071393e+000; foe(n+1)=5.898519e+000; krok(n+1)=9.469203e-001; ng(n+1)=3.283861e+002;
n=25; farx(n+1)=9.051086e-001; foe(n+1)=5.316177e+000; krok(n+1)=9.872820e-001; ng(n+1)=1.731847e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=9.081179e-001; foe(n+1)=5.283492e+000; krok(n+1)=3.828771e-005; ng(n+1)=9.528373e+001;
n=27; farx(n+1)=9.025046e-001; foe(n+1)=5.266990e+000; krok(n+1)=5.234435e-005; ng(n+1)=6.007707e+001;
n=28; farx(n+1)=8.754903e-001; foe(n+1)=5.181220e+000; krok(n+1)=2.625683e-004; ng(n+1)=5.562884e+001;
n=29; farx(n+1)=9.118106e-001; foe(n+1)=5.076751e+000; krok(n+1)=4.312903e-003; ng(n+1)=1.550317e+001;
n=30; farx(n+1)=9.357269e-001; foe(n+1)=4.817109e+000; krok(n+1)=2.578794e-002; ng(n+1)=2.148681e+001;
n=31; farx(n+1)=8.855056e-001; foe(n+1)=4.661356e+000; krok(n+1)=3.586741e-002; ng(n+1)=1.135411e+002;
n=32; farx(n+1)=8.145121e-001; foe(n+1)=4.299005e+000; krok(n+1)=4.703297e-002; ng(n+1)=1.446470e+002;
n=33; farx(n+1)=7.857126e-001; foe(n+1)=4.080192e+000; krok(n+1)=1.251384e-002; ng(n+1)=2.834699e+002;
n=34; farx(n+1)=7.697619e-001; foe(n+1)=4.047898e+000; krok(n+1)=2.307159e-002; ng(n+1)=2.380316e+002;
n=35; farx(n+1)=6.343572e-001; foe(n+1)=3.653144e+000; krok(n+1)=1.803262e-002; ng(n+1)=2.635183e+002;
n=36; farx(n+1)=5.311499e-001; foe(n+1)=1.915415e+000; krok(n+1)=3.550835e-001; ng(n+1)=2.240293e+002;
n=37; farx(n+1)=4.620005e-001; foe(n+1)=1.657792e+000; krok(n+1)=9.729904e-002; ng(n+1)=9.715471e+001;
n=38; farx(n+1)=4.709445e-001; foe(n+1)=1.563769e+000; krok(n+1)=1.481341e-001; ng(n+1)=1.881483e+002;
n=39; farx(n+1)=4.741240e-001; foe(n+1)=1.510698e+000; krok(n+1)=8.100548e-002; ng(n+1)=1.364217e+002;
n=40; farx(n+1)=4.792893e-001; foe(n+1)=1.438629e+000; krok(n+1)=3.963116e-001; ng(n+1)=1.579132e+002;
n=41; farx(n+1)=5.167027e-001; foe(n+1)=1.239687e+000; krok(n+1)=9.362720e-001; ng(n+1)=1.880505e+002;
n=42; farx(n+1)=4.424131e-001; foe(n+1)=1.018648e+000; krok(n+1)=7.113726e-001; ng(n+1)=6.518139e+001;
n=43; farx(n+1)=4.194687e-001; foe(n+1)=9.100215e-001; krok(n+1)=1.732520e-001; ng(n+1)=1.501659e+002;
n=44; farx(n+1)=4.064850e-001; foe(n+1)=7.540575e-001; krok(n+1)=7.853583e-002; ng(n+1)=2.530855e+002;
n=45; farx(n+1)=3.849165e-001; foe(n+1)=6.819994e-001; krok(n+1)=4.880844e-001; ng(n+1)=4.967777e+001;
n=46; farx(n+1)=3.464989e-001; foe(n+1)=6.240045e-001; krok(n+1)=1.691553e+000; ng(n+1)=6.433803e+001;
n=47; farx(n+1)=2.972040e-001; foe(n+1)=5.494339e-001; krok(n+1)=1.445753e+000; ng(n+1)=3.909589e+001;
n=48; farx(n+1)=2.872739e-001; foe(n+1)=5.370300e-001; krok(n+1)=6.167589e-001; ng(n+1)=2.911803e+001;
n=49; farx(n+1)=2.821571e-001; foe(n+1)=5.282863e-001; krok(n+1)=2.363480e-001; ng(n+1)=3.403274e+001;
n=50; farx(n+1)=2.683614e-001; foe(n+1)=5.219917e-001; krok(n+1)=6.522451e-001; ng(n+1)=1.762175e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
