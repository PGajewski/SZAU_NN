%uczenie predyktora oe
clear all;
n=0; farx(n+1)=1.813417e+002; foe(n+1)=1.833860e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.576354e+002; foe(n+1)=1.629136e+002; krok(n+1)=6.399708e-004; ng(n+1)=5.955440e+002;
n=2; farx(n+1)=5.445798e+001; foe(n+1)=6.743189e+001; krok(n+1)=4.904166e-003; ng(n+1)=4.531154e+002;
n=3; farx(n+1)=4.869370e+001; foe(n+1)=5.975175e+001; krok(n+1)=1.460933e-003; ng(n+1)=3.996116e+002;
n=4; farx(n+1)=4.571828e+001; foe(n+1)=5.878720e+001; krok(n+1)=4.587536e-003; ng(n+1)=8.357532e+001;
n=5; farx(n+1)=1.274868e+001; foe(n+1)=4.643389e+001; krok(n+1)=2.502130e-002; ng(n+1)=1.236218e+002;
n=6; farx(n+1)=7.201820e+000; foe(n+1)=4.305880e+001; krok(n+1)=4.751309e-004; ng(n+1)=1.342645e+003;
n=7; farx(n+1)=5.305039e+000; foe(n+1)=4.146690e+001; krok(n+1)=5.469456e-004; ng(n+1)=2.442366e+003;
n=8; farx(n+1)=4.283814e+000; foe(n+1)=3.895132e+001; krok(n+1)=6.650130e-004; ng(n+1)=3.890934e+003;
n=9; farx(n+1)=4.045237e+000; foe(n+1)=3.778038e+001; krok(n+1)=2.307756e-003; ng(n+1)=5.424705e+003;
n=10; farx(n+1)=3.944665e+000; foe(n+1)=3.733914e+001; krok(n+1)=8.940659e-004; ng(n+1)=5.369238e+003;
n=11; farx(n+1)=3.137717e+000; foe(n+1)=3.135127e+001; krok(n+1)=2.067140e-002; ng(n+1)=4.807681e+003;
n=12; farx(n+1)=3.388180e+000; foe(n+1)=2.997532e+001; krok(n+1)=7.131570e-005; ng(n+1)=3.937551e+003;
n=13; farx(n+1)=3.694231e+000; foe(n+1)=2.927603e+001; krok(n+1)=1.222281e-003; ng(n+1)=3.162744e+003;
n=14; farx(n+1)=5.473796e+000; foe(n+1)=1.950417e+001; krok(n+1)=6.255325e-003; ng(n+1)=3.047815e+003;
n=15; farx(n+1)=5.921718e+000; foe(n+1)=1.847995e+001; krok(n+1)=2.716773e-003; ng(n+1)=1.129566e+003;
n=16; farx(n+1)=5.108460e+000; foe(n+1)=1.749251e+001; krok(n+1)=1.601914e-002; ng(n+1)=4.623621e+002;
n=17; farx(n+1)=4.231851e+000; foe(n+1)=1.615309e+001; krok(n+1)=1.251065e-002; ng(n+1)=2.126465e+002;
n=18; farx(n+1)=3.609887e+000; foe(n+1)=1.516172e+001; krok(n+1)=2.570124e-002; ng(n+1)=4.506023e+002;
n=19; farx(n+1)=3.220168e+000; foe(n+1)=1.449583e+001; krok(n+1)=4.938447e-003; ng(n+1)=3.343707e+002;
n=20; farx(n+1)=2.415631e+000; foe(n+1)=1.305300e+001; krok(n+1)=5.344961e-002; ng(n+1)=1.911277e+002;
n=21; farx(n+1)=2.004166e+000; foe(n+1)=1.204545e+001; krok(n+1)=1.034736e-002; ng(n+1)=5.436030e+002;
n=22; farx(n+1)=2.023702e+000; foe(n+1)=1.136303e+001; krok(n+1)=6.254878e-003; ng(n+1)=2.426025e+002;
n=23; farx(n+1)=2.035416e+000; foe(n+1)=1.125200e+001; krok(n+1)=5.608324e-003; ng(n+1)=5.237315e+002;
n=24; farx(n+1)=2.212603e+000; foe(n+1)=1.077515e+001; krok(n+1)=8.877087e-002; ng(n+1)=4.234230e+002;
n=25; farx(n+1)=2.404915e+000; foe(n+1)=1.027295e+001; krok(n+1)=1.360507e-002; ng(n+1)=6.458351e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=2.435519e+000; foe(n+1)=1.016643e+001; krok(n+1)=4.356664e-004; ng(n+1)=7.627538e+001;
n=27; farx(n+1)=2.429363e+000; foe(n+1)=1.015820e+001; krok(n+1)=3.351833e-005; ng(n+1)=6.661294e+001;
n=28; farx(n+1)=2.418861e+000; foe(n+1)=1.013311e+001; krok(n+1)=1.231773e-004; ng(n+1)=6.406941e+001;
n=29; farx(n+1)=1.836963e+000; foe(n+1)=9.352773e+000; krok(n+1)=2.818678e-003; ng(n+1)=7.286613e+001;
n=30; farx(n+1)=1.359081e+000; foe(n+1)=8.353181e+000; krok(n+1)=1.012183e-002; ng(n+1)=1.087075e+002;
n=31; farx(n+1)=1.232046e+000; foe(n+1)=8.035239e+000; krok(n+1)=1.171923e-003; ng(n+1)=4.677392e+002;
n=32; farx(n+1)=1.065790e+000; foe(n+1)=7.293188e+000; krok(n+1)=1.835014e-002; ng(n+1)=3.168624e+002;
n=33; farx(n+1)=1.038304e+000; foe(n+1)=6.783162e+000; krok(n+1)=5.915258e-003; ng(n+1)=5.750926e+002;
n=34; farx(n+1)=1.077220e+000; foe(n+1)=6.002953e+000; krok(n+1)=1.573351e-002; ng(n+1)=2.886007e+002;
n=35; farx(n+1)=1.042064e+000; foe(n+1)=5.209047e+000; krok(n+1)=1.076348e-002; ng(n+1)=1.287808e+002;
n=36; farx(n+1)=1.035252e+000; foe(n+1)=4.797145e+000; krok(n+1)=3.835857e-003; ng(n+1)=2.502855e+002;
n=37; farx(n+1)=1.023149e+000; foe(n+1)=4.552063e+000; krok(n+1)=7.899388e-004; ng(n+1)=5.376073e+002;
n=38; farx(n+1)=1.034989e+000; foe(n+1)=3.759733e+000; krok(n+1)=7.557177e-003; ng(n+1)=2.562440e+002;
n=39; farx(n+1)=1.012199e+000; foe(n+1)=3.441573e+000; krok(n+1)=1.034736e-002; ng(n+1)=5.777824e+002;
n=40; farx(n+1)=1.033182e+000; foe(n+1)=3.264348e+000; krok(n+1)=4.049572e-003; ng(n+1)=3.249237e+002;
n=41; farx(n+1)=1.012276e+000; foe(n+1)=2.924387e+000; krok(n+1)=1.725161e-002; ng(n+1)=3.883202e+002;
n=42; farx(n+1)=1.025452e+000; foe(n+1)=2.565542e+000; krok(n+1)=1.340015e-002; ng(n+1)=5.613113e+002;
n=43; farx(n+1)=9.640213e-001; foe(n+1)=2.122275e+000; krok(n+1)=6.485153e-003; ng(n+1)=2.281416e+002;
n=44; farx(n+1)=8.986018e-001; foe(n+1)=1.763670e+000; krok(n+1)=1.064021e-002; ng(n+1)=4.375617e+002;
n=45; farx(n+1)=8.055231e-001; foe(n+1)=1.561552e+000; krok(n+1)=2.387143e-002; ng(n+1)=5.083232e+002;
n=46; farx(n+1)=7.798121e-001; foe(n+1)=1.503004e+000; krok(n+1)=2.667694e-002; ng(n+1)=2.932338e+002;
n=47; farx(n+1)=7.554093e-001; foe(n+1)=1.478468e+000; krok(n+1)=3.413297e-002; ng(n+1)=4.614445e+001;
n=48; farx(n+1)=6.468854e-001; foe(n+1)=1.403418e+000; krok(n+1)=3.809184e-002; ng(n+1)=1.305583e+002;
n=49; farx(n+1)=6.347766e-001; foe(n+1)=1.388605e+000; krok(n+1)=3.482091e-002; ng(n+1)=1.549746e+002;
n=50; farx(n+1)=5.662632e-001; foe(n+1)=1.299614e+000; krok(n+1)=1.423178e-001; ng(n+1)=5.740300e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
