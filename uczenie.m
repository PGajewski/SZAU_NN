%uczenie predyktora oe
clear all;
n=0; farx(n+1)=1.780174e+002; foe(n+1)=1.875676e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.458337e+002; foe(n+1)=1.540967e+002; krok(n+1)=6.459813e-004; ng(n+1)=9.420896e+002;
n=2; farx(n+1)=5.640329e+001; foe(n+1)=6.165513e+001; krok(n+1)=2.972972e-003; ng(n+1)=7.429652e+002;
n=3; farx(n+1)=5.300047e+001; foe(n+1)=5.612926e+001; krok(n+1)=1.838554e-003; ng(n+1)=3.729707e+002;
n=4; farx(n+1)=4.840319e+001; foe(n+1)=5.517558e+001; krok(n+1)=5.303205e-003; ng(n+1)=8.503390e+001;
n=5; farx(n+1)=1.946204e+001; foe(n+1)=4.255343e+001; krok(n+1)=3.280985e-002; ng(n+1)=1.221691e+002;
n=6; farx(n+1)=1.076946e+001; foe(n+1)=3.715125e+001; krok(n+1)=3.581841e-003; ng(n+1)=7.032462e+002;
n=7; farx(n+1)=7.246071e+000; foe(n+1)=2.618577e+001; krok(n+1)=2.914778e-003; ng(n+1)=1.473808e+003;
n=8; farx(n+1)=7.315276e+000; foe(n+1)=2.565768e+001; krok(n+1)=4.334515e-005; ng(n+1)=2.660956e+003;
n=9; farx(n+1)=7.319041e+000; foe(n+1)=2.515321e+001; krok(n+1)=2.336140e-003; ng(n+1)=2.366954e+003;
n=10; farx(n+1)=8.305280e+000; foe(n+1)=2.244582e+001; krok(n+1)=1.025657e-002; ng(n+1)=2.487821e+003;
n=11; farx(n+1)=8.776312e+000; foe(n+1)=1.851585e+001; krok(n+1)=1.667309e-003; ng(n+1)=1.808242e+003;
n=12; farx(n+1)=8.731126e+000; foe(n+1)=1.657753e+001; krok(n+1)=2.032662e-003; ng(n+1)=7.314094e+002;
n=13; farx(n+1)=7.987338e+000; foe(n+1)=1.570688e+001; krok(n+1)=1.438335e-002; ng(n+1)=3.160412e+002;
n=14; farx(n+1)=7.130090e+000; foe(n+1)=1.493406e+001; krok(n+1)=1.809047e-002; ng(n+1)=1.597591e+002;
n=15; farx(n+1)=4.950983e+000; foe(n+1)=1.282145e+001; krok(n+1)=1.820180e-002; ng(n+1)=2.827226e+002;
n=16; farx(n+1)=3.945242e+000; foe(n+1)=1.190129e+001; krok(n+1)=8.128862e-003; ng(n+1)=2.417301e+002;
n=17; farx(n+1)=1.673536e+000; foe(n+1)=8.583002e+000; krok(n+1)=1.000726e-002; ng(n+1)=3.092663e+002;
n=18; farx(n+1)=1.367312e+000; foe(n+1)=7.404485e+000; krok(n+1)=6.223487e-003; ng(n+1)=3.008880e+002;
n=19; farx(n+1)=1.265289e+000; foe(n+1)=6.513293e+000; krok(n+1)=8.684522e-004; ng(n+1)=4.533922e+002;
n=20; farx(n+1)=1.129434e+000; foe(n+1)=5.909643e+000; krok(n+1)=1.309024e-002; ng(n+1)=5.338624e+002;
n=21; farx(n+1)=1.042524e+000; foe(n+1)=5.588038e+000; krok(n+1)=3.242576e-003; ng(n+1)=2.953175e+002;
n=22; farx(n+1)=1.057479e+000; foe(n+1)=5.301067e+000; krok(n+1)=2.531421e-003; ng(n+1)=1.614864e+002;
n=23; farx(n+1)=1.050980e+000; foe(n+1)=5.065681e+000; krok(n+1)=9.195867e-003; ng(n+1)=2.277698e+002;
n=24; farx(n+1)=7.942941e-001; foe(n+1)=4.257200e+000; krok(n+1)=6.640781e-002; ng(n+1)=1.001247e+002;
n=25; farx(n+1)=7.127849e-001; foe(n+1)=3.861869e+000; krok(n+1)=5.095665e-003; ng(n+1)=2.909049e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=7.077501e-001; foe(n+1)=3.589716e+000; krok(n+1)=3.190024e-005; ng(n+1)=3.706508e+002;
n=27; farx(n+1)=7.164843e-001; foe(n+1)=3.560422e+000; krok(n+1)=2.963075e-005; ng(n+1)=1.101674e+002;
n=28; farx(n+1)=7.191554e-001; foe(n+1)=3.546809e+000; krok(n+1)=1.581535e-004; ng(n+1)=4.183374e+001;
n=29; farx(n+1)=6.093439e-001; foe(n+1)=3.185633e+000; krok(n+1)=1.430505e-002; ng(n+1)=2.795302e+001;
n=30; farx(n+1)=5.655575e-001; foe(n+1)=2.776184e+000; krok(n+1)=6.543079e-003; ng(n+1)=9.860881e+001;
n=31; farx(n+1)=5.309518e-001; foe(n+1)=2.397648e+000; krok(n+1)=1.345452e-002; ng(n+1)=1.916960e+002;
n=32; farx(n+1)=5.157599e-001; foe(n+1)=2.153105e+000; krok(n+1)=2.778799e-003; ng(n+1)=2.439643e+002;
n=33; farx(n+1)=4.817035e-001; foe(n+1)=1.760104e+000; krok(n+1)=1.012183e-002; ng(n+1)=1.950294e+002;
n=34; farx(n+1)=4.979161e-001; foe(n+1)=1.487187e+000; krok(n+1)=1.417782e-002; ng(n+1)=2.670451e+002;
n=35; farx(n+1)=4.856785e-001; foe(n+1)=1.325985e+000; krok(n+1)=1.153579e-002; ng(n+1)=1.992936e+002;
n=36; farx(n+1)=4.700402e-001; foe(n+1)=1.226097e+000; krok(n+1)=2.156300e-002; ng(n+1)=1.357493e+002;
n=37; farx(n+1)=4.700482e-001; foe(n+1)=1.191727e+000; krok(n+1)=9.175071e-003; ng(n+1)=9.574737e+001;
n=38; farx(n+1)=4.658057e-001; foe(n+1)=1.173928e+000; krok(n+1)=6.963555e-003; ng(n+1)=7.747562e+001;
n=39; farx(n+1)=4.657507e-001; foe(n+1)=1.106366e+000; krok(n+1)=4.000365e-002; ng(n+1)=3.751662e+001;
n=40; farx(n+1)=4.603860e-001; foe(n+1)=1.051607e+000; krok(n+1)=4.841816e-003; ng(n+1)=7.522557e+001;
n=41; farx(n+1)=4.617253e-001; foe(n+1)=1.026551e+000; krok(n+1)=2.173419e-002; ng(n+1)=7.984502e+001;
n=42; farx(n+1)=4.198553e-001; foe(n+1)=9.450380e-001; krok(n+1)=3.680593e-002; ng(n+1)=9.072510e+001;
n=43; farx(n+1)=3.984870e-001; foe(n+1)=8.936939e-001; krok(n+1)=3.612031e-002; ng(n+1)=1.249606e+002;
n=44; farx(n+1)=3.735639e-001; foe(n+1)=8.558060e-001; krok(n+1)=1.726831e-002; ng(n+1)=1.007388e+002;
n=45; farx(n+1)=3.653244e-001; foe(n+1)=8.310344e-001; krok(n+1)=5.440189e-002; ng(n+1)=9.740644e+001;
n=46; farx(n+1)=3.606280e-001; foe(n+1)=7.477912e-001; krok(n+1)=2.283902e-001; ng(n+1)=6.873260e+001;
n=47; farx(n+1)=3.413274e-001; foe(n+1)=7.195433e-001; krok(n+1)=3.459424e-002; ng(n+1)=5.568576e+001;
n=48; farx(n+1)=2.989784e-001; foe(n+1)=6.638651e-001; krok(n+1)=8.349282e-003; ng(n+1)=1.394968e+002;
n=49; farx(n+1)=2.951579e-001; foe(n+1)=6.593989e-001; krok(n+1)=6.422675e-003; ng(n+1)=5.991639e+001;
n=50; farx(n+1)=2.932156e-001; foe(n+1)=6.383513e-001; krok(n+1)=4.215364e-002; ng(n+1)=6.732984e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)