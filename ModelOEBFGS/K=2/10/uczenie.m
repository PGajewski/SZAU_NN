%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.897973e+002; foe(n+1)=2.869595e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.771082e+002; foe(n+1)=1.745878e+002; krok(n+1)=5.033545e-004; ng(n+1)=6.971071e+002;
n=2; farx(n+1)=6.696081e+001; foe(n+1)=6.460078e+001; krok(n+1)=1.925105e-002; ng(n+1)=1.201893e+002;
n=3; farx(n+1)=6.194975e+001; foe(n+1)=5.791727e+001; krok(n+1)=1.189189e-002; ng(n+1)=1.625056e+002;
n=4; farx(n+1)=5.875451e+001; foe(n+1)=5.724271e+001; krok(n+1)=8.202462e-003; ng(n+1)=5.531446e+001;
n=5; farx(n+1)=3.894708e+001; foe(n+1)=5.095807e+001; krok(n+1)=1.767909e-002; ng(n+1)=7.472145e+001;
n=6; farx(n+1)=3.178926e+001; foe(n+1)=4.734055e+001; krok(n+1)=4.568872e-002; ng(n+1)=1.497155e+002;
n=7; farx(n+1)=3.146970e+001; foe(n+1)=4.656737e+001; krok(n+1)=1.363555e-002; ng(n+1)=9.654554e+001;
n=8; farx(n+1)=2.896809e+001; foe(n+1)=4.549178e+001; krok(n+1)=4.215364e-002; ng(n+1)=1.231528e+002;
n=9; farx(n+1)=1.860910e+001; foe(n+1)=4.217135e+001; krok(n+1)=3.769903e-002; ng(n+1)=1.374361e+002;
n=10; farx(n+1)=8.577195e+000; foe(n+1)=3.312902e+001; krok(n+1)=2.504009e-001; ng(n+1)=2.484784e+002;
n=11; farx(n+1)=1.343109e+001; foe(n+1)=2.927925e+001; krok(n+1)=3.450079e-001; ng(n+1)=3.169448e+002;
n=12; farx(n+1)=1.173583e+001; foe(n+1)=2.713280e+001; krok(n+1)=3.655097e-001; ng(n+1)=1.547761e+002;
n=13; farx(n+1)=6.867826e+000; foe(n+1)=2.278775e+001; krok(n+1)=3.805404e-001; ng(n+1)=1.159813e+002;
n=14; farx(n+1)=5.870955e+000; foe(n+1)=2.224839e+001; krok(n+1)=1.619492e-001; ng(n+1)=1.135456e+002;
n=15; farx(n+1)=2.649502e+000; foe(n+1)=1.789316e+001; krok(n+1)=1.474115e-001; ng(n+1)=2.014334e+002;
n=16; farx(n+1)=2.529091e+000; foe(n+1)=1.546143e+001; krok(n+1)=5.801457e-002; ng(n+1)=5.447097e+002;
n=17; farx(n+1)=2.692253e+000; foe(n+1)=1.421612e+001; krok(n+1)=3.276344e-002; ng(n+1)=5.250802e+002;
n=18; farx(n+1)=2.730581e+000; foe(n+1)=1.406088e+001; krok(n+1)=5.128285e-003; ng(n+1)=6.548470e+002;
n=19; farx(n+1)=3.348215e+000; foe(n+1)=1.109920e+001; krok(n+1)=3.307424e-001; ng(n+1)=4.270342e+002;
n=20; farx(n+1)=3.320811e+000; foe(n+1)=9.916505e+000; krok(n+1)=3.862514e-001; ng(n+1)=2.092104e+002;
n=21; farx(n+1)=3.207663e+000; foe(n+1)=9.579986e+000; krok(n+1)=4.938403e-001; ng(n+1)=6.068519e+001;
n=22; farx(n+1)=2.862053e+000; foe(n+1)=8.876077e+000; krok(n+1)=3.903951e-001; ng(n+1)=2.541882e+002;
n=23; farx(n+1)=2.014538e+000; foe(n+1)=7.772248e+000; krok(n+1)=2.697833e+000; ng(n+1)=1.008068e+002;
n=24; farx(n+1)=2.201290e+000; foe(n+1)=7.150885e+000; krok(n+1)=9.861549e-001; ng(n+1)=8.893893e+001;
n=25; farx(n+1)=2.304455e+000; foe(n+1)=6.797157e+000; krok(n+1)=5.174513e-001; ng(n+1)=1.856802e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=2.310608e+000; foe(n+1)=6.708862e+000; krok(n+1)=1.197513e-005; ng(n+1)=1.635885e+002;
n=27; farx(n+1)=2.333424e+000; foe(n+1)=6.696563e+000; krok(n+1)=1.219597e-004; ng(n+1)=2.539837e+001;
n=28; farx(n+1)=2.360009e+000; foe(n+1)=6.673447e+000; krok(n+1)=6.467102e-004; ng(n+1)=1.669296e+001;
n=29; farx(n+1)=2.310791e+000; foe(n+1)=6.463233e+000; krok(n+1)=2.586841e-003; ng(n+1)=2.938613e+001;
n=30; farx(n+1)=2.163173e+000; foe(n+1)=6.202184e+000; krok(n+1)=1.830045e-002; ng(n+1)=1.980685e+001;
n=31; farx(n+1)=2.186963e+000; foe(n+1)=5.983545e+000; krok(n+1)=7.876901e-003; ng(n+1)=1.239573e+002;
n=32; farx(n+1)=2.075456e+000; foe(n+1)=5.655054e+000; krok(n+1)=1.530377e-001; ng(n+1)=6.772726e+001;
n=33; farx(n+1)=1.557972e+000; foe(n+1)=4.345318e+000; krok(n+1)=3.655097e-001; ng(n+1)=9.707208e+001;
n=34; farx(n+1)=1.501487e+000; foe(n+1)=4.116303e+000; krok(n+1)=1.355729e-001; ng(n+1)=1.427271e+002;
n=35; farx(n+1)=1.548867e+000; foe(n+1)=3.729809e+000; krok(n+1)=3.768236e-001; ng(n+1)=4.040688e+001;
n=36; farx(n+1)=1.462016e+000; foe(n+1)=3.566532e+000; krok(n+1)=2.475890e-001; ng(n+1)=2.057653e+001;
n=37; farx(n+1)=1.269459e+000; foe(n+1)=3.334270e+000; krok(n+1)=8.707243e-001; ng(n+1)=2.368780e+001;
n=38; farx(n+1)=1.171954e+000; foe(n+1)=3.226126e+000; krok(n+1)=5.491644e-001; ng(n+1)=7.521199e+001;
n=39; farx(n+1)=1.144994e+000; foe(n+1)=3.134523e+000; krok(n+1)=1.086288e+000; ng(n+1)=4.308002e+001;
n=40; farx(n+1)=1.103140e+000; foe(n+1)=3.067428e+000; krok(n+1)=4.556164e-001; ng(n+1)=2.588533e+001;
n=41; farx(n+1)=1.068531e+000; foe(n+1)=3.029590e+000; krok(n+1)=5.154350e-001; ng(n+1)=3.068705e+001;
n=42; farx(n+1)=8.805993e-001; foe(n+1)=2.868091e+000; krok(n+1)=4.061290e+000; ng(n+1)=2.853262e+001;
n=43; farx(n+1)=8.187652e-001; foe(n+1)=2.769688e+000; krok(n+1)=7.273018e-001; ng(n+1)=4.028953e+001;
n=44; farx(n+1)=7.818784e-001; foe(n+1)=2.654208e+000; krok(n+1)=3.963116e-001; ng(n+1)=8.837280e+001;
n=45; farx(n+1)=7.494392e-001; foe(n+1)=2.622886e+000; krok(n+1)=1.582278e-001; ng(n+1)=6.715695e+001;
n=46; farx(n+1)=7.097212e-001; foe(n+1)=2.600264e+000; krok(n+1)=8.904025e-001; ng(n+1)=1.641483e+001;
n=47; farx(n+1)=6.739365e-001; foe(n+1)=2.589314e+000; krok(n+1)=8.375141e-001; ng(n+1)=2.943253e+001;
n=48; farx(n+1)=6.615841e-001; foe(n+1)=2.580781e+000; krok(n+1)=9.001093e-001; ng(n+1)=1.238437e+001;
n=49; farx(n+1)=5.853814e-001; foe(n+1)=2.547973e+000; krok(n+1)=2.477523e+000; ng(n+1)=6.683934e+000;
n=50; farx(n+1)=5.643211e-001; foe(n+1)=2.534916e+000; krok(n+1)=7.638857e-001; ng(n+1)=1.277378e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
