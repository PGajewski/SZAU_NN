%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.367437e+002; foe(n+1)=2.369490e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.793665e+002; foe(n+1)=1.794648e+002; krok(n+1)=5.202484e-004; ng(n+1)=1.094573e+003;
n=2; farx(n+1)=6.055035e+001; foe(n+1)=6.820850e+001; krok(n+1)=1.048565e-002; ng(n+1)=4.921714e+002;
n=3; farx(n+1)=5.045771e+001; foe(n+1)=5.986282e+001; krok(n+1)=3.729504e-004; ng(n+1)=9.125319e+002;
n=4; farx(n+1)=4.791281e+001; foe(n+1)=5.917288e+001; krok(n+1)=1.212360e-003; ng(n+1)=1.521341e+002;
n=5; farx(n+1)=1.656597e+001; foe(n+1)=5.069455e+001; krok(n+1)=9.661282e-003; ng(n+1)=1.544238e+002;
n=6; farx(n+1)=4.129917e+000; foe(n+1)=3.690033e+001; krok(n+1)=6.538025e-003; ng(n+1)=1.504472e+003;
n=7; farx(n+1)=3.783782e+000; foe(n+1)=3.634424e+001; krok(n+1)=2.167257e-005; ng(n+1)=6.736114e+003;
n=8; farx(n+1)=3.457756e+000; foe(n+1)=3.557150e+001; krok(n+1)=1.425029e-003; ng(n+1)=7.525983e+003;
n=9; farx(n+1)=4.244997e+000; foe(n+1)=2.559567e+001; krok(n+1)=7.637224e-003; ng(n+1)=8.322383e+003;
n=10; farx(n+1)=4.923445e+000; foe(n+1)=2.425400e+001; krok(n+1)=1.802468e-004; ng(n+1)=4.478123e+003;
n=11; farx(n+1)=6.190757e+000; foe(n+1)=2.090833e+001; krok(n+1)=2.677949e-003; ng(n+1)=3.243651e+003;
n=12; farx(n+1)=6.256397e+000; foe(n+1)=1.820875e+001; krok(n+1)=1.540867e-002; ng(n+1)=9.062919e+002;
n=13; farx(n+1)=6.410814e+000; foe(n+1)=1.730482e+001; krok(n+1)=9.748751e-003; ng(n+1)=4.356287e+002;
n=14; farx(n+1)=6.161572e+000; foe(n+1)=1.573262e+001; krok(n+1)=2.857938e-002; ng(n+1)=2.602744e+002;
n=15; farx(n+1)=5.951291e+000; foe(n+1)=1.481972e+001; krok(n+1)=2.957629e-003; ng(n+1)=4.522185e+002;
n=16; farx(n+1)=5.886914e+000; foe(n+1)=1.463425e+001; krok(n+1)=1.603465e-003; ng(n+1)=2.669471e+002;
n=17; farx(n+1)=4.093196e+000; foe(n+1)=1.027763e+001; krok(n+1)=3.500452e-002; ng(n+1)=3.430392e+002;
n=18; farx(n+1)=3.877021e+000; foe(n+1)=9.931458e+000; krok(n+1)=6.174461e-004; ng(n+1)=9.620811e+002;
n=19; farx(n+1)=3.419718e+000; foe(n+1)=9.349643e+000; krok(n+1)=4.375565e-003; ng(n+1)=5.264729e+002;
n=20; farx(n+1)=2.930577e+000; foe(n+1)=8.504896e+000; krok(n+1)=4.653602e-003; ng(n+1)=5.803376e+002;
n=21; farx(n+1)=2.513787e+000; foe(n+1)=7.754331e+000; krok(n+1)=2.502768e-002; ng(n+1)=4.042035e+002;
n=22; farx(n+1)=2.347265e+000; foe(n+1)=7.548694e+000; krok(n+1)=2.331589e-003; ng(n+1)=4.600093e+002;
n=23; farx(n+1)=1.966270e+000; foe(n+1)=6.738808e+000; krok(n+1)=5.038654e-003; ng(n+1)=5.942537e+002;
n=24; farx(n+1)=1.829468e+000; foe(n+1)=6.149263e+000; krok(n+1)=5.770731e-003; ng(n+1)=4.402208e+002;
n=25; farx(n+1)=1.532432e+000; foe(n+1)=4.660315e+000; krok(n+1)=9.921120e-003; ng(n+1)=6.284842e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.434444e+000; foe(n+1)=4.145656e+000; krok(n+1)=2.581106e-005; ng(n+1)=6.249847e+002;
n=27; farx(n+1)=1.409833e+000; foe(n+1)=4.094038e+000; krok(n+1)=2.039081e-004; ng(n+1)=8.199713e+001;
n=28; farx(n+1)=1.361086e+000; foe(n+1)=3.995834e+000; krok(n+1)=1.729605e-005; ng(n+1)=3.284117e+002;
n=29; farx(n+1)=1.299838e+000; foe(n+1)=3.742746e+000; krok(n+1)=6.037139e-004; ng(n+1)=9.491569e+001;
n=30; farx(n+1)=1.074203e+000; foe(n+1)=2.961726e+000; krok(n+1)=7.525650e-003; ng(n+1)=5.788082e+001;
n=31; farx(n+1)=9.659320e-001; foe(n+1)=2.330056e+000; krok(n+1)=1.831590e-002; ng(n+1)=2.650974e+002;
n=32; farx(n+1)=9.435302e-001; foe(n+1)=2.171050e+000; krok(n+1)=1.046070e-003; ng(n+1)=5.332042e+002;
n=33; farx(n+1)=8.951872e-001; foe(n+1)=1.920642e+000; krok(n+1)=4.290347e-003; ng(n+1)=2.852693e+002;
n=34; farx(n+1)=8.708629e-001; foe(n+1)=1.862283e+000; krok(n+1)=1.165911e-002; ng(n+1)=1.321040e+002;
n=35; farx(n+1)=8.077314e-001; foe(n+1)=1.736244e+000; krok(n+1)=1.664795e-002; ng(n+1)=2.741706e+002;
n=36; farx(n+1)=7.322415e-001; foe(n+1)=1.588771e+000; krok(n+1)=2.928765e-002; ng(n+1)=1.847962e+002;
n=37; farx(n+1)=6.159657e-001; foe(n+1)=1.301510e+000; krok(n+1)=2.024365e-002; ng(n+1)=4.208092e+002;
n=38; farx(n+1)=6.120377e-001; foe(n+1)=1.262806e+000; krok(n+1)=4.550450e-003; ng(n+1)=2.199237e+002;
n=39; farx(n+1)=6.023998e-001; foe(n+1)=1.228353e+000; krok(n+1)=7.431661e-003; ng(n+1)=1.144943e+002;
n=40; farx(n+1)=5.768318e-001; foe(n+1)=1.181713e+000; krok(n+1)=1.023953e-002; ng(n+1)=1.298617e+002;
n=41; farx(n+1)=5.537319e-001; foe(n+1)=1.110436e+000; krok(n+1)=4.346837e-002; ng(n+1)=1.685034e+002;
n=42; farx(n+1)=5.473101e-001; foe(n+1)=1.086269e+000; krok(n+1)=2.000182e-002; ng(n+1)=1.773106e+002;
n=43; farx(n+1)=5.311793e-001; foe(n+1)=1.026625e+000; krok(n+1)=4.880471e-002; ng(n+1)=7.632423e+001;
n=44; farx(n+1)=5.299248e-001; foe(n+1)=1.003305e+000; krok(n+1)=2.811708e-002; ng(n+1)=7.436128e+001;
n=45; farx(n+1)=5.257392e-001; foe(n+1)=9.849614e-001; krok(n+1)=4.438543e-002; ng(n+1)=1.338914e+002;
n=46; farx(n+1)=4.872111e-001; foe(n+1)=9.630202e-001; krok(n+1)=1.380129e-001; ng(n+1)=2.144492e+001;
n=47; farx(n+1)=4.602044e-001; foe(n+1)=9.462981e-001; krok(n+1)=8.316649e-002; ng(n+1)=6.908136e+001;
n=48; farx(n+1)=4.433423e-001; foe(n+1)=9.356888e-001; krok(n+1)=2.093303e-002; ng(n+1)=8.454233e+001;
n=49; farx(n+1)=4.259617e-001; foe(n+1)=9.259383e-001; krok(n+1)=1.205272e-002; ng(n+1)=5.605559e+001;
n=50; farx(n+1)=4.025742e-001; foe(n+1)=8.905845e-001; krok(n+1)=1.271551e-001; ng(n+1)=5.022083e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)