%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.688310e+002; foe(n+1)=2.768579e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.512072e+002; foe(n+1)=1.566328e+002; krok(n+1)=4.744783e-004; ng(n+1)=9.336816e+002;
n=2; farx(n+1)=6.189547e+001; foe(n+1)=6.916079e+001; krok(n+1)=6.353015e-003; ng(n+1)=2.797150e+002;
n=3; farx(n+1)=5.789110e+001; foe(n+1)=6.263650e+001; krok(n+1)=2.094724e-003; ng(n+1)=3.148744e+002;
n=4; farx(n+1)=5.891850e+001; foe(n+1)=6.084433e+001; krok(n+1)=1.280341e-002; ng(n+1)=1.804841e+002;
n=5; farx(n+1)=4.398256e+001; foe(n+1)=5.793418e+001; krok(n+1)=2.378798e-002; ng(n+1)=3.264507e+001;
n=6; farx(n+1)=2.691136e+001; foe(n+1)=5.403230e+001; krok(n+1)=1.547817e-002; ng(n+1)=2.340974e+002;
n=7; farx(n+1)=1.913799e+001; foe(n+1)=5.269447e+001; krok(n+1)=6.459813e-004; ng(n+1)=4.696481e+002;
n=8; farx(n+1)=1.172793e+001; foe(n+1)=4.708087e+001; krok(n+1)=3.139181e-002; ng(n+1)=7.918396e+002;
n=9; farx(n+1)=1.082836e+001; foe(n+1)=4.656865e+001; krok(n+1)=1.401071e-004; ng(n+1)=1.078752e+003;
n=10; farx(n+1)=8.955371e+000; foe(n+1)=4.416798e+001; krok(n+1)=1.815661e-002; ng(n+1)=1.365160e+003;
n=11; farx(n+1)=7.500667e+000; foe(n+1)=4.082091e+001; krok(n+1)=2.818159e-003; ng(n+1)=1.583744e+003;
n=12; farx(n+1)=7.079420e+000; foe(n+1)=4.008410e+001; krok(n+1)=8.456028e-004; ng(n+1)=1.470214e+003;
n=13; farx(n+1)=6.534187e+000; foe(n+1)=3.639685e+001; krok(n+1)=1.559274e-002; ng(n+1)=1.561821e+003;
n=14; farx(n+1)=6.518401e+000; foe(n+1)=2.972254e+001; krok(n+1)=4.427789e-003; ng(n+1)=1.604656e+003;
n=15; farx(n+1)=7.327311e+000; foe(n+1)=2.417792e+001; krok(n+1)=3.569431e-004; ng(n+1)=1.747066e+003;
n=16; farx(n+1)=7.614168e+000; foe(n+1)=2.261184e+001; krok(n+1)=6.192151e-004; ng(n+1)=1.046311e+003;
n=17; farx(n+1)=7.695888e+000; foe(n+1)=2.210932e+001; krok(n+1)=3.013180e-003; ng(n+1)=4.117604e+002;
n=18; farx(n+1)=7.551077e+000; foe(n+1)=2.053289e+001; krok(n+1)=4.048730e-002; ng(n+1)=2.570637e+002;
n=19; farx(n+1)=6.748668e+000; foe(n+1)=1.782688e+001; krok(n+1)=8.625198e-002; ng(n+1)=2.338054e+002;
n=20; farx(n+1)=5.547093e+000; foe(n+1)=1.614990e+001; krok(n+1)=1.746541e-001; ng(n+1)=2.159055e+002;
n=21; farx(n+1)=3.287398e+000; foe(n+1)=1.223866e+001; krok(n+1)=7.382907e-001; ng(n+1)=1.437672e+002;
n=22; farx(n+1)=2.824339e+000; foe(n+1)=1.078121e+001; krok(n+1)=9.760941e-002; ng(n+1)=2.691146e+002;
n=23; farx(n+1)=1.652333e+000; foe(n+1)=8.958671e+000; krok(n+1)=6.537835e-001; ng(n+1)=2.088554e+002;
n=24; farx(n+1)=1.552603e+000; foe(n+1)=8.734908e+000; krok(n+1)=1.210679e-001; ng(n+1)=1.556551e+002;
n=25; farx(n+1)=1.463356e+000; foe(n+1)=8.502591e+000; krok(n+1)=2.857025e-001; ng(n+1)=1.800626e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.460591e+000; foe(n+1)=8.493831e+000; krok(n+1)=1.765974e-005; ng(n+1)=6.628192e+001;
n=27; farx(n+1)=1.425116e+000; foe(n+1)=8.461906e+000; krok(n+1)=1.802468e-004; ng(n+1)=3.749039e+001;
n=28; farx(n+1)=1.423033e+000; foe(n+1)=8.453716e+000; krok(n+1)=3.559311e-004; ng(n+1)=1.682512e+001;
n=29; farx(n+1)=1.427940e+000; foe(n+1)=8.315351e+000; krok(n+1)=1.323510e-003; ng(n+1)=3.896802e+001;
n=30; farx(n+1)=1.365454e+000; foe(n+1)=8.079348e+000; krok(n+1)=2.107682e-002; ng(n+1)=1.584086e+001;
n=31; farx(n+1)=1.325363e+000; foe(n+1)=7.826273e+000; krok(n+1)=2.490870e-002; ng(n+1)=1.686559e+002;
n=32; farx(n+1)=1.117049e+000; foe(n+1)=7.245549e+000; krok(n+1)=1.254246e-002; ng(n+1)=1.381333e+002;
n=33; farx(n+1)=1.069596e+000; foe(n+1)=7.015116e+000; krok(n+1)=4.952046e-003; ng(n+1)=1.489830e+002;
n=34; farx(n+1)=1.031807e+000; foe(n+1)=6.722662e+000; krok(n+1)=1.997747e-003; ng(n+1)=4.071646e+002;
n=35; farx(n+1)=9.951072e-001; foe(n+1)=6.398927e+000; krok(n+1)=5.062842e-003; ng(n+1)=4.466247e+002;
n=36; farx(n+1)=9.710855e-001; foe(n+1)=6.206786e+000; krok(n+1)=2.428721e-002; ng(n+1)=7.588255e+002;
n=37; farx(n+1)=1.077012e+000; foe(n+1)=5.771733e+000; krok(n+1)=3.559503e-003; ng(n+1)=9.328942e+002;
n=38; farx(n+1)=1.148799e+000; foe(n+1)=5.628676e+000; krok(n+1)=2.972972e-003; ng(n+1)=1.106101e+003;
n=39; farx(n+1)=1.226730e+000; foe(n+1)=5.167958e+000; krok(n+1)=1.028050e-001; ng(n+1)=1.025140e+003;
n=40; farx(n+1)=1.278708e+000; foe(n+1)=4.643106e+000; krok(n+1)=9.809225e-002; ng(n+1)=7.815445e+002;
n=41; farx(n+1)=1.297737e+000; foe(n+1)=4.192821e+000; krok(n+1)=5.375048e-002; ng(n+1)=4.790697e+002;
n=42; farx(n+1)=1.282337e+000; foe(n+1)=4.066412e+000; krok(n+1)=2.194150e-002; ng(n+1)=2.629695e+002;
n=43; farx(n+1)=1.162294e+000; foe(n+1)=3.866559e+000; krok(n+1)=1.981558e-001; ng(n+1)=5.227267e+001;
n=44; farx(n+1)=1.032113e+000; foe(n+1)=3.472868e+000; krok(n+1)=2.613267e-001; ng(n+1)=2.407255e+002;
n=45; farx(n+1)=8.858598e-001; foe(n+1)=3.329296e+000; krok(n+1)=8.008859e-001; ng(n+1)=1.556189e+002;
n=46; farx(n+1)=8.246908e-001; foe(n+1)=3.114657e+000; krok(n+1)=1.223081e+000; ng(n+1)=9.019491e+001;
n=47; farx(n+1)=7.592933e-001; foe(n+1)=2.953289e+000; krok(n+1)=5.203615e-001; ng(n+1)=1.521984e+002;
n=48; farx(n+1)=7.567505e-001; foe(n+1)=2.800032e+000; krok(n+1)=7.610808e-001; ng(n+1)=7.718409e+001;
n=49; farx(n+1)=7.559858e-001; foe(n+1)=2.695083e+000; krok(n+1)=3.140763e-001; ng(n+1)=1.915327e+002;
n=50; farx(n+1)=7.380254e-001; foe(n+1)=2.657908e+000; krok(n+1)=3.589327e-001; ng(n+1)=6.621245e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
