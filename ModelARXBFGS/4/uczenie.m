%uczenie predyktora arx
clear all;
n=0; farx(n+1)=2.486835e+002; foe(n+1)=2.471862e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.823629e+002; foe(n+1)=1.849819e+002; krok(n+1)=5.014459e-004; ng(n+1)=8.189608e+002;
n=2; farx(n+1)=1.646337e+002; foe(n+1)=1.754066e+002; krok(n+1)=4.648167e-003; ng(n+1)=2.489796e+002;
n=3; farx(n+1)=2.825705e+001; foe(n+1)=6.201344e+001; krok(n+1)=4.541426e-003; ng(n+1)=3.773060e+002;
n=4; farx(n+1)=2.361339e+001; foe(n+1)=5.794819e+001; krok(n+1)=1.882699e-003; ng(n+1)=2.501973e+002;
n=5; farx(n+1)=1.449733e+001; foe(n+1)=1.339839e+002; krok(n+1)=1.012568e-002; ng(n+1)=2.055651e+002;
n=6; farx(n+1)=6.764437e+000; foe(n+1)=1.090614e+003; krok(n+1)=1.111520e-002; ng(n+1)=5.087957e+002;
n=7; farx(n+1)=5.870793e+000; foe(n+1)=1.405276e+003; krok(n+1)=8.202462e-003; ng(n+1)=2.405357e+002;
n=8; farx(n+1)=2.900638e+000; foe(n+1)=5.484638e+002; krok(n+1)=1.141951e-001; ng(n+1)=1.007800e+002;
n=9; farx(n+1)=2.377714e+000; foe(n+1)=3.495104e+002; krok(n+1)=1.598198e-002; ng(n+1)=1.050681e+002;
n=10; farx(n+1)=1.541034e+000; foe(n+1)=1.386288e+002; krok(n+1)=5.793922e-002; ng(n+1)=9.493292e+001;
n=11; farx(n+1)=1.390474e+000; foe(n+1)=1.151826e+002; krok(n+1)=5.135894e-002; ng(n+1)=4.610346e+001;
n=12; farx(n+1)=1.027634e+000; foe(n+1)=9.556745e+001; krok(n+1)=6.827250e-002; ng(n+1)=4.352235e+001;
n=13; farx(n+1)=9.450582e-001; foe(n+1)=3.811922e+001; krok(n+1)=9.809225e-002; ng(n+1)=4.551863e+001;
n=14; farx(n+1)=8.307454e-001; foe(n+1)=1.962900e+001; krok(n+1)=7.424094e-002; ng(n+1)=5.925809e+001;
n=15; farx(n+1)=5.720050e-001; foe(n+1)=2.208961e+001; krok(n+1)=6.709302e-001; ng(n+1)=2.422299e+001;
n=16; farx(n+1)=4.469170e-001; foe(n+1)=1.430649e+001; krok(n+1)=7.310195e-001; ng(n+1)=1.846500e+001;
n=17; farx(n+1)=3.801439e-001; foe(n+1)=7.122100e+000; krok(n+1)=2.249366e-001; ng(n+1)=3.099217e+001;
n=18; farx(n+1)=3.341346e-001; foe(n+1)=3.446845e+000; krok(n+1)=3.240219e-001; ng(n+1)=2.735492e+001;
n=19; farx(n+1)=3.245698e-001; foe(n+1)=3.208999e+000; krok(n+1)=3.032584e-001; ng(n+1)=1.026771e+001;
n=20; farx(n+1)=3.101151e-001; foe(n+1)=3.080152e+000; krok(n+1)=9.169512e-001; ng(n+1)=4.670929e+000;
n=21; farx(n+1)=2.837243e-001; foe(n+1)=2.150149e+000; krok(n+1)=1.040723e+000; ng(n+1)=8.965322e+000;
n=22; farx(n+1)=2.578874e-001; foe(n+1)=3.997191e+000; krok(n+1)=4.635137e-001; ng(n+1)=5.901601e+000;
n=23; farx(n+1)=2.474524e-001; foe(n+1)=2.906279e+000; krok(n+1)=3.743784e-001; ng(n+1)=7.590296e+000;
n=24; farx(n+1)=2.430711e-001; foe(n+1)=2.188345e+000; krok(n+1)=8.008859e-001; ng(n+1)=5.549896e+000;
n=25; farx(n+1)=2.407720e-001; foe(n+1)=2.251786e+000; krok(n+1)=1.462039e+000; ng(n+1)=1.651862e+000;
%odnowa zmiennej metryki
n=26; farx(n+1)=2.398797e-001; foe(n+1)=2.190237e+000; krok(n+1)=5.333277e-004; ng(n+1)=4.746539e+000;
n=27; farx(n+1)=2.396281e-001; foe(n+1)=2.451091e+000; krok(n+1)=2.165649e-002; ng(n+1)=4.069098e-001;
n=28; farx(n+1)=2.370224e-001; foe(n+1)=2.234398e+000; krok(n+1)=8.268560e-002; ng(n+1)=7.268950e-001;
n=29; farx(n+1)=2.362743e-001; foe(n+1)=1.881032e+000; krok(n+1)=3.276764e-004; ng(n+1)=4.347817e+000;
n=30; farx(n+1)=2.336227e-001; foe(n+1)=1.757763e+000; krok(n+1)=9.963479e-002; ng(n+1)=1.042008e+000;
n=31; farx(n+1)=2.322223e-001; foe(n+1)=1.578101e+000; krok(n+1)=2.710777e-001; ng(n+1)=3.796042e+000;
n=32; farx(n+1)=2.284882e-001; foe(n+1)=1.476168e+000; krok(n+1)=1.177071e+000; ng(n+1)=5.016397e+000;
n=33; farx(n+1)=2.268002e-001; foe(n+1)=1.730886e+000; krok(n+1)=3.068907e-001; ng(n+1)=3.750837e+000;
n=34; farx(n+1)=2.214841e-001; foe(n+1)=1.504287e+000; krok(n+1)=8.067975e-001; ng(n+1)=3.290790e+000;
n=35; farx(n+1)=2.204725e-001; foe(n+1)=1.232300e+000; krok(n+1)=1.652088e-001; ng(n+1)=5.260782e+000;
n=36; farx(n+1)=2.185798e-001; foe(n+1)=1.142107e+000; krok(n+1)=3.217882e-001; ng(n+1)=4.480729e+000;
n=37; farx(n+1)=2.176761e-001; foe(n+1)=1.183380e+000; krok(n+1)=2.935451e-001; ng(n+1)=1.047621e+000;
n=38; farx(n+1)=2.163283e-001; foe(n+1)=1.108493e+000; krok(n+1)=5.690454e-001; ng(n+1)=1.542666e+000;
n=39; farx(n+1)=2.146316e-001; foe(n+1)=1.248505e+000; krok(n+1)=5.636152e-001; ng(n+1)=3.971941e+000;
n=40; farx(n+1)=2.138246e-001; foe(n+1)=1.192832e+000; krok(n+1)=1.083912e+000; ng(n+1)=1.547168e+000;
n=41; farx(n+1)=2.132170e-001; foe(n+1)=1.066163e+000; krok(n+1)=7.676757e-001; ng(n+1)=1.536192e+000;
n=42; farx(n+1)=2.128751e-001; foe(n+1)=1.078037e+000; krok(n+1)=5.570755e-001; ng(n+1)=1.171820e+000;
n=43; farx(n+1)=2.125229e-001; foe(n+1)=9.952919e-001; krok(n+1)=4.419445e-001; ng(n+1)=2.605946e+000;
n=44; farx(n+1)=2.121115e-001; foe(n+1)=9.644212e-001; krok(n+1)=1.304892e+000; ng(n+1)=5.520539e-001;
n=45; farx(n+1)=2.111927e-001; foe(n+1)=1.062015e+000; krok(n+1)=2.699639e+000; ng(n+1)=1.312002e+000;
n=46; farx(n+1)=2.102243e-001; foe(n+1)=1.024724e+000; krok(n+1)=1.215405e+000; ng(n+1)=2.880126e+000;
n=47; farx(n+1)=2.092567e-001; foe(n+1)=9.951539e-001; krok(n+1)=5.045340e-001; ng(n+1)=2.204662e+000;
n=48; farx(n+1)=2.086267e-001; foe(n+1)=9.714782e-001; krok(n+1)=3.885953e-001; ng(n+1)=2.869580e+000;
n=49; farx(n+1)=2.080995e-001; foe(n+1)=8.756667e-001; krok(n+1)=7.386536e-001; ng(n+1)=3.615252e+000;
n=50; farx(n+1)=2.079979e-001; foe(n+1)=9.049588e-001; krok(n+1)=1.752988e-001; ng(n+1)=7.008787e-001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora ARX');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
