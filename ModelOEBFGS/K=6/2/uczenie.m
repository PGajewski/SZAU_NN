%uczenie predyktora oe
clear all;
n=0; farx(n+1)=1.936632e+002; foe(n+1)=1.910219e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.754718e+002; foe(n+1)=1.727153e+002; krok(n+1)=6.187216e-004; ng(n+1)=4.821670e+002;
n=2; farx(n+1)=6.870797e+001; foe(n+1)=6.220158e+001; krok(n+1)=9.939766e-003; ng(n+1)=3.319063e+002;
n=3; farx(n+1)=6.737670e+001; foe(n+1)=6.131670e+001; krok(n+1)=9.704157e-004; ng(n+1)=1.310335e+002;
n=4; farx(n+1)=5.677447e+001; foe(n+1)=6.013398e+001; krok(n+1)=6.374691e-003; ng(n+1)=5.926329e+001;
n=5; farx(n+1)=5.550583e+000; foe(n+1)=2.992004e+001; krok(n+1)=1.041486e-002; ng(n+1)=1.653655e+002;
n=6; farx(n+1)=4.871511e+000; foe(n+1)=2.943731e+001; krok(n+1)=3.251553e-005; ng(n+1)=1.862740e+003;
n=7; farx(n+1)=5.014810e+000; foe(n+1)=2.923883e+001; krok(n+1)=2.397584e-003; ng(n+1)=2.302996e+003;
n=8; farx(n+1)=5.644297e+000; foe(n+1)=1.578677e+001; krok(n+1)=2.502130e-002; ng(n+1)=2.292472e+003;
n=9; farx(n+1)=5.638487e+000; foe(n+1)=1.538195e+001; krok(n+1)=2.040296e-003; ng(n+1)=2.542303e+002;
n=10; farx(n+1)=5.551496e+000; foe(n+1)=1.516366e+001; krok(n+1)=1.670282e-003; ng(n+1)=1.754404e+002;
n=11; farx(n+1)=4.952720e+000; foe(n+1)=1.426210e+001; krok(n+1)=2.346492e-003; ng(n+1)=2.060541e+002;
n=12; farx(n+1)=4.082328e+000; foe(n+1)=1.304483e+001; krok(n+1)=3.094863e-002; ng(n+1)=2.514392e+002;
n=13; farx(n+1)=3.085781e+000; foe(n+1)=1.025829e+001; krok(n+1)=3.218320e-003; ng(n+1)=7.297562e+002;
n=14; farx(n+1)=2.757175e+000; foe(n+1)=9.682234e+000; krok(n+1)=1.118600e-003; ng(n+1)=6.215177e+002;
n=15; farx(n+1)=2.144725e+000; foe(n+1)=7.922119e+000; krok(n+1)=1.023953e-002; ng(n+1)=7.315301e+002;
n=16; farx(n+1)=1.962359e+000; foe(n+1)=7.439039e+000; krok(n+1)=7.029269e-003; ng(n+1)=7.124397e+002;
n=17; farx(n+1)=1.492508e+000; foe(n+1)=6.119366e+000; krok(n+1)=1.740861e-002; ng(n+1)=4.032469e+002;
n=18; farx(n+1)=1.243835e+000; foe(n+1)=5.654449e+000; krok(n+1)=6.319511e-003; ng(n+1)=3.981031e+002;
n=19; farx(n+1)=8.982788e-001; foe(n+1)=4.801476e+000; krok(n+1)=9.866494e-003; ng(n+1)=1.740033e+002;
n=20; farx(n+1)=8.337948e-001; foe(n+1)=4.661913e+000; krok(n+1)=6.669234e-003; ng(n+1)=3.162291e+002;
n=21; farx(n+1)=7.270489e-001; foe(n+1)=4.166943e+000; krok(n+1)=3.329740e-002; ng(n+1)=1.856386e+002;
n=22; farx(n+1)=6.987269e-001; foe(n+1)=3.878706e+000; krok(n+1)=3.221469e-002; ng(n+1)=2.141881e+002;
n=23; farx(n+1)=6.553854e-001; foe(n+1)=3.661575e+000; krok(n+1)=2.320653e-002; ng(n+1)=2.015772e+002;
n=24; farx(n+1)=5.846018e-001; foe(n+1)=2.912926e+000; krok(n+1)=8.005809e-002; ng(n+1)=2.089787e+002;
n=25; farx(n+1)=5.153619e-001; foe(n+1)=2.409801e+000; krok(n+1)=6.423218e-002; ng(n+1)=3.529419e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=5.134015e-001; foe(n+1)=2.256896e+000; krok(n+1)=1.399412e-005; ng(n+1)=3.437904e+002;
n=27; farx(n+1)=5.159848e-001; foe(n+1)=2.163833e+000; krok(n+1)=1.527405e-005; ng(n+1)=3.046422e+002;
n=28; farx(n+1)=4.916154e-001; foe(n+1)=2.074057e+000; krok(n+1)=3.271540e-003; ng(n+1)=1.952728e+001;
n=29; farx(n+1)=4.702013e-001; foe(n+1)=1.955914e+000; krok(n+1)=4.108016e-004; ng(n+1)=6.220852e+001;
n=30; farx(n+1)=4.701450e-001; foe(n+1)=1.539507e+000; krok(n+1)=1.214548e-002; ng(n+1)=3.442122e+001;
n=31; farx(n+1)=4.597157e-001; foe(n+1)=1.451961e+000; krok(n+1)=6.905382e-003; ng(n+1)=1.602588e+002;
n=32; farx(n+1)=4.698298e-001; foe(n+1)=1.382521e+000; krok(n+1)=9.335866e-003; ng(n+1)=1.628485e+002;
n=33; farx(n+1)=5.230710e-001; foe(n+1)=1.270176e+000; krok(n+1)=1.025657e-002; ng(n+1)=8.785759e+001;
n=34; farx(n+1)=5.600323e-001; foe(n+1)=1.181216e+000; krok(n+1)=3.251545e-002; ng(n+1)=8.492572e+001;
n=35; farx(n+1)=5.166752e-001; foe(n+1)=1.088719e+000; krok(n+1)=3.178877e-002; ng(n+1)=3.857167e+001;
n=36; farx(n+1)=5.094074e-001; foe(n+1)=1.069894e+000; krok(n+1)=5.915528e-003; ng(n+1)=1.028999e+002;
n=37; farx(n+1)=4.983795e-001; foe(n+1)=9.723014e-001; krok(n+1)=5.851700e-002; ng(n+1)=7.816178e+001;
n=38; farx(n+1)=4.995420e-001; foe(n+1)=9.583651e-001; krok(n+1)=1.540867e-002; ng(n+1)=4.786126e+001;
n=39; farx(n+1)=4.952015e-001; foe(n+1)=9.355139e-001; krok(n+1)=2.835564e-002; ng(n+1)=3.299855e+001;
n=40; farx(n+1)=4.911016e-001; foe(n+1)=9.230133e-001; krok(n+1)=8.662598e-002; ng(n+1)=6.430920e+001;
n=41; farx(n+1)=4.891567e-001; foe(n+1)=9.182176e-001; krok(n+1)=1.660195e-002; ng(n+1)=5.892417e+001;
n=42; farx(n+1)=4.582403e-001; foe(n+1)=9.010671e-001; krok(n+1)=2.701916e-001; ng(n+1)=7.844136e+000;
n=43; farx(n+1)=4.416433e-001; foe(n+1)=8.914072e-001; krok(n+1)=1.417782e-002; ng(n+1)=6.807865e+001;
n=44; farx(n+1)=4.336571e-001; foe(n+1)=8.801213e-001; krok(n+1)=6.864555e-002; ng(n+1)=3.366875e+001;
n=45; farx(n+1)=4.311325e-001; foe(n+1)=8.734925e-001; krok(n+1)=3.153338e-002; ng(n+1)=4.784322e+001;
n=46; farx(n+1)=4.281201e-001; foe(n+1)=8.606001e-001; krok(n+1)=1.420813e-001; ng(n+1)=7.837723e+001;
n=47; farx(n+1)=4.115286e-001; foe(n+1)=8.403629e-001; krok(n+1)=1.492580e-001; ng(n+1)=4.693268e+001;
n=48; farx(n+1)=3.884707e-001; foe(n+1)=8.024713e-001; krok(n+1)=3.457398e-001; ng(n+1)=1.061729e+002;
n=49; farx(n+1)=3.795209e-001; foe(n+1)=7.874993e-001; krok(n+1)=1.291716e-001; ng(n+1)=5.535578e+001;
n=50; farx(n+1)=3.653761e-001; foe(n+1)=7.738937e-001; krok(n+1)=2.032965e-001; ng(n+1)=3.818001e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)