%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.934276e+002; foe(n+1)=2.923690e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.768631e+002; foe(n+1)=1.764358e+002; krok(n+1)=5.033545e-004; ng(n+1)=5.840674e+002;
n=2; farx(n+1)=6.719895e+001; foe(n+1)=6.789113e+001; krok(n+1)=2.690904e-002; ng(n+1)=7.322799e+001;
n=3; farx(n+1)=6.575066e+001; foe(n+1)=6.625845e+001; krok(n+1)=3.293253e-004; ng(n+1)=2.633660e+002;
n=4; farx(n+1)=7.443417e+001; foe(n+1)=6.232853e+001; krok(n+1)=1.238474e-002; ng(n+1)=3.175391e+002;
n=5; farx(n+1)=6.286158e+001; foe(n+1)=5.989319e+001; krok(n+1)=1.737095e-002; ng(n+1)=5.273433e+001;
n=6; farx(n+1)=4.034743e+001; foe(n+1)=5.017786e+001; krok(n+1)=5.466235e-002; ng(n+1)=2.178492e+002;
n=7; farx(n+1)=3.905476e+001; foe(n+1)=4.898790e+001; krok(n+1)=3.715831e-003; ng(n+1)=4.114261e+002;
n=8; farx(n+1)=3.554569e+001; foe(n+1)=4.553777e+001; krok(n+1)=1.366559e-002; ng(n+1)=3.821912e+002;
n=9; farx(n+1)=2.203718e+001; foe(n+1)=3.730751e+001; krok(n+1)=1.484819e-001; ng(n+1)=3.074585e+002;
n=10; farx(n+1)=1.995626e+001; foe(n+1)=3.657275e+001; krok(n+1)=2.728971e-003; ng(n+1)=2.054662e+002;
n=11; farx(n+1)=1.582874e+001; foe(n+1)=3.441347e+001; krok(n+1)=1.193571e-002; ng(n+1)=2.469236e+002;
n=12; farx(n+1)=1.395616e+001; foe(n+1)=3.268662e+001; krok(n+1)=1.031514e-003; ng(n+1)=3.988538e+002;
n=13; farx(n+1)=1.274226e+001; foe(n+1)=3.054449e+001; krok(n+1)=9.454744e-003; ng(n+1)=5.297399e+002;
n=14; farx(n+1)=1.276026e+001; foe(n+1)=3.029881e+001; krok(n+1)=8.443488e-003; ng(n+1)=2.883872e+002;
n=15; farx(n+1)=1.022805e+001; foe(n+1)=2.709183e+001; krok(n+1)=1.331896e-001; ng(n+1)=2.601770e+002;
n=16; farx(n+1)=4.906404e+000; foe(n+1)=2.057066e+001; krok(n+1)=1.601162e-001; ng(n+1)=2.458637e+002;
n=17; farx(n+1)=4.609104e+000; foe(n+1)=1.900006e+001; krok(n+1)=6.004768e-002; ng(n+1)=4.719390e+002;
n=18; farx(n+1)=2.677775e+000; foe(n+1)=1.379839e+001; krok(n+1)=2.317569e-001; ng(n+1)=1.007381e+002;
n=19; farx(n+1)=2.884643e+000; foe(n+1)=1.239111e+001; krok(n+1)=7.406706e-002; ng(n+1)=3.390208e+002;
n=20; farx(n+1)=2.045738e+000; foe(n+1)=1.069390e+001; krok(n+1)=1.755320e-001; ng(n+1)=1.437686e+002;
n=21; farx(n+1)=9.768212e-001; foe(n+1)=8.522802e+000; krok(n+1)=5.821767e-001; ng(n+1)=4.095546e+002;
n=22; farx(n+1)=8.723464e-001; foe(n+1)=7.806661e+000; krok(n+1)=3.550835e-001; ng(n+1)=1.025885e+002;
n=23; farx(n+1)=8.886478e-001; foe(n+1)=7.513522e+000; krok(n+1)=2.047709e-001; ng(n+1)=1.238264e+002;
n=24; farx(n+1)=9.255596e-001; foe(n+1)=6.888925e+000; krok(n+1)=7.676757e-001; ng(n+1)=2.223865e+002;
n=25; farx(n+1)=9.159371e-001; foe(n+1)=6.738988e+000; krok(n+1)=3.428611e-001; ng(n+1)=1.965551e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=9.151022e-001; foe(n+1)=6.732812e+000; krok(n+1)=7.048309e-006; ng(n+1)=6.532973e+001;
n=27; farx(n+1)=9.179187e-001; foe(n+1)=6.723502e+000; krok(n+1)=1.445574e-004; ng(n+1)=2.220106e+001;
n=28; farx(n+1)=9.305006e-001; foe(n+1)=6.592951e+000; krok(n+1)=6.752731e-004; ng(n+1)=5.203280e+001;
n=29; farx(n+1)=8.802424e-001; foe(n+1)=6.271542e+000; krok(n+1)=4.624551e-004; ng(n+1)=7.696054e+001;
n=30; farx(n+1)=8.171363e-001; foe(n+1)=5.982135e+000; krok(n+1)=3.752980e-003; ng(n+1)=9.190586e+001;
n=31; farx(n+1)=7.860427e-001; foe(n+1)=5.248478e+000; krok(n+1)=8.193332e-003; ng(n+1)=4.460390e+002;
n=32; farx(n+1)=7.903524e-001; foe(n+1)=5.106518e+000; krok(n+1)=3.904632e-003; ng(n+1)=2.507469e+002;
n=33; farx(n+1)=7.765964e-001; foe(n+1)=4.637559e+000; krok(n+1)=1.059691e-002; ng(n+1)=3.914083e+002;
n=34; farx(n+1)=8.017983e-001; foe(n+1)=4.363553e+000; krok(n+1)=1.021537e-002; ng(n+1)=4.990981e+002;
n=35; farx(n+1)=8.560191e-001; foe(n+1)=4.264287e+000; krok(n+1)=1.417782e-002; ng(n+1)=4.369465e+002;
n=36; farx(n+1)=9.938498e-001; foe(n+1)=4.048016e+000; krok(n+1)=6.105720e-002; ng(n+1)=3.235875e+002;
n=37; farx(n+1)=1.302084e+000; foe(n+1)=3.319797e+000; krok(n+1)=1.827549e-001; ng(n+1)=1.560854e+002;
n=38; farx(n+1)=1.155874e+000; foe(n+1)=3.064883e+000; krok(n+1)=6.384007e-002; ng(n+1)=1.276867e+002;
n=39; farx(n+1)=1.062658e+000; foe(n+1)=2.939505e+000; krok(n+1)=9.569438e-002; ng(n+1)=9.904626e+001;
n=40; farx(n+1)=9.598973e-001; foe(n+1)=2.773688e+000; krok(n+1)=1.600146e-001; ng(n+1)=3.240625e+002;
n=41; farx(n+1)=7.288319e-001; foe(n+1)=2.468995e+000; krok(n+1)=8.973318e-002; ng(n+1)=2.489140e+002;
n=42; farx(n+1)=6.806145e-001; foe(n+1)=2.327789e+000; krok(n+1)=2.723382e-001; ng(n+1)=9.581693e+001;
n=43; farx(n+1)=5.942745e-001; foe(n+1)=2.043914e+000; krok(n+1)=1.224301e+000; ng(n+1)=1.111829e+002;
n=44; farx(n+1)=5.467591e-001; foe(n+1)=1.864874e+000; krok(n+1)=3.885953e-001; ng(n+1)=2.904557e+002;
n=45; farx(n+1)=4.837046e-001; foe(n+1)=1.710462e+000; krok(n+1)=6.762233e-001; ng(n+1)=7.175899e+001;
n=46; farx(n+1)=4.145188e-001; foe(n+1)=1.567920e+000; krok(n+1)=4.079802e-001; ng(n+1)=9.382453e+001;
n=47; farx(n+1)=4.177825e-001; foe(n+1)=1.484071e+000; krok(n+1)=9.270274e-001; ng(n+1)=3.655154e+001;
n=48; farx(n+1)=4.468642e-001; foe(n+1)=1.356168e+000; krok(n+1)=1.133514e+000; ng(n+1)=8.981111e+001;
n=49; farx(n+1)=4.683269e-001; foe(n+1)=1.302378e+000; krok(n+1)=8.771846e-001; ng(n+1)=6.518443e+001;
n=50; farx(n+1)=4.860955e-001; foe(n+1)=1.245592e+000; krok(n+1)=1.223081e+000; ng(n+1)=1.219095e+002;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
