%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.223218e+002; foe(n+1)=2.246515e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.753146e+002; foe(n+1)=1.770632e+002; krok(n+1)=5.045766e-004; ng(n+1)=8.738429e+002;
n=2; farx(n+1)=7.245378e+001; foe(n+1)=6.076949e+001; krok(n+1)=1.791379e-002; ng(n+1)=3.904377e+002;
n=3; farx(n+1)=6.845552e+001; foe(n+1)=5.878653e+001; krok(n+1)=5.683150e-004; ng(n+1)=3.475454e+002;
n=4; farx(n+1)=6.600084e+001; foe(n+1)=5.802167e+001; krok(n+1)=9.846126e-004; ng(n+1)=1.493413e+002;
n=5; farx(n+1)=1.471067e+001; foe(n+1)=4.193339e+001; krok(n+1)=1.617524e-002; ng(n+1)=1.487947e+002;
n=6; farx(n+1)=7.822783e+000; foe(n+1)=3.803598e+001; krok(n+1)=1.293420e-003; ng(n+1)=1.206626e+003;
n=7; farx(n+1)=6.062720e+000; foe(n+1)=3.077603e+001; krok(n+1)=1.631264e-003; ng(n+1)=2.729789e+003;
n=8; farx(n+1)=6.166613e+000; foe(n+1)=3.031333e+001; krok(n+1)=1.854565e-003; ng(n+1)=3.326217e+003;
n=9; farx(n+1)=8.035622e+000; foe(n+1)=2.793695e+001; krok(n+1)=1.464382e-002; ng(n+1)=2.847529e+003;
n=10; farx(n+1)=8.699858e+000; foe(n+1)=2.570306e+001; krok(n+1)=8.898758e-004; ng(n+1)=1.342577e+003;
n=11; farx(n+1)=9.142176e+000; foe(n+1)=2.451631e+001; krok(n+1)=2.501815e-003; ng(n+1)=7.028962e+002;
n=12; farx(n+1)=9.086477e+000; foe(n+1)=2.350575e+001; krok(n+1)=6.399091e-003; ng(n+1)=3.509947e+002;
n=13; farx(n+1)=9.323471e+000; foe(n+1)=2.214809e+001; krok(n+1)=1.005588e-002; ng(n+1)=6.643343e+002;
n=14; farx(n+1)=9.064372e+000; foe(n+1)=2.146113e+001; krok(n+1)=4.175393e-003; ng(n+1)=3.226655e+002;
n=15; farx(n+1)=7.351837e+000; foe(n+1)=1.904075e+001; krok(n+1)=1.706649e-002; ng(n+1)=2.993662e+002;
n=16; farx(n+1)=6.854953e+000; foe(n+1)=1.849033e+001; krok(n+1)=2.433134e-003; ng(n+1)=8.388185e+002;
n=17; farx(n+1)=4.947952e+000; foe(n+1)=1.407088e+001; krok(n+1)=1.464382e-002; ng(n+1)=9.336115e+002;
n=18; farx(n+1)=4.409176e+000; foe(n+1)=1.334503e+001; krok(n+1)=3.128460e-003; ng(n+1)=2.976775e+002;
n=19; farx(n+1)=3.423891e+000; foe(n+1)=1.117471e+001; krok(n+1)=4.403244e-003; ng(n+1)=7.668540e+002;
n=20; farx(n+1)=3.127883e+000; foe(n+1)=1.037966e+001; krok(n+1)=1.532691e-003; ng(n+1)=5.001543e+002;
n=21; farx(n+1)=2.783282e+000; foe(n+1)=9.247407e+000; krok(n+1)=6.255325e-003; ng(n+1)=4.370955e+002;
n=22; farx(n+1)=2.375223e+000; foe(n+1)=7.728302e+000; krok(n+1)=5.981703e-003; ng(n+1)=3.276946e+002;
n=23; farx(n+1)=2.123027e+000; foe(n+1)=7.218435e+000; krok(n+1)=2.855545e-003; ng(n+1)=4.563677e+002;
n=24; farx(n+1)=1.908109e+000; foe(n+1)=6.316390e+000; krok(n+1)=2.716773e-003; ng(n+1)=4.251225e+002;
n=25; farx(n+1)=1.731849e+000; foe(n+1)=5.430899e+000; krok(n+1)=7.242402e-003; ng(n+1)=4.665761e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.608747e+000; foe(n+1)=4.824941e+000; krok(n+1)=2.907853e-005; ng(n+1)=4.866984e+002;
n=27; farx(n+1)=1.613687e+000; foe(n+1)=4.774728e+000; krok(n+1)=2.560827e-005; ng(n+1)=1.854310e+002;
n=28; farx(n+1)=1.511927e+000; foe(n+1)=4.345779e+000; krok(n+1)=9.272824e-004; ng(n+1)=9.969628e+001;
n=29; farx(n+1)=1.310779e+000; foe(n+1)=3.563942e+000; krok(n+1)=7.394073e-004; ng(n+1)=1.641473e+002;
n=30; farx(n+1)=1.136883e+000; foe(n+1)=2.682240e+000; krok(n+1)=3.881663e-003; ng(n+1)=6.908372e+001;
n=31; farx(n+1)=1.030692e+000; foe(n+1)=2.019940e+000; krok(n+1)=5.627734e-003; ng(n+1)=3.735682e+002;
n=32; farx(n+1)=9.820205e-001; foe(n+1)=1.529772e+000; krok(n+1)=6.681202e-003; ng(n+1)=3.271577e+002;
n=33; farx(n+1)=8.711560e-001; foe(n+1)=1.347107e+000; krok(n+1)=2.340953e-002; ng(n+1)=8.292331e+001;
n=34; farx(n+1)=7.945007e-001; foe(n+1)=1.228218e+000; krok(n+1)=9.683632e-003; ng(n+1)=1.544199e+002;
n=35; farx(n+1)=7.252372e-001; foe(n+1)=1.154093e+000; krok(n+1)=1.238474e-002; ng(n+1)=1.109210e+002;
n=36; farx(n+1)=6.040369e-001; foe(n+1)=1.006337e+000; krok(n+1)=1.183106e-002; ng(n+1)=1.916795e+002;
n=37; farx(n+1)=5.580759e-001; foe(n+1)=9.482499e-001; krok(n+1)=4.218186e-002; ng(n+1)=5.636221e+001;
n=38; farx(n+1)=5.494639e-001; foe(n+1)=9.359159e-001; krok(n+1)=6.854056e-003; ng(n+1)=9.718759e+001;
n=39; farx(n+1)=5.308370e-001; foe(n+1)=9.097887e-001; krok(n+1)=3.684354e-002; ng(n+1)=3.789768e+001;
n=40; farx(n+1)=5.157131e-001; foe(n+1)=8.839585e-001; krok(n+1)=5.004260e-002; ng(n+1)=3.851421e+001;
n=41; farx(n+1)=4.920800e-001; foe(n+1)=8.366729e-001; krok(n+1)=7.610620e-002; ng(n+1)=9.209224e+001;
n=42; farx(n+1)=4.866618e-001; foe(n+1)=8.246292e-001; krok(n+1)=6.446985e-003; ng(n+1)=1.732509e+002;
n=43; farx(n+1)=4.846048e-001; foe(n+1)=8.072730e-001; krok(n+1)=4.517978e-002; ng(n+1)=9.707623e+001;
n=44; farx(n+1)=4.822439e-001; foe(n+1)=7.891866e-001; krok(n+1)=7.808311e-002; ng(n+1)=5.939353e+001;
n=45; farx(n+1)=4.878459e-001; foe(n+1)=7.783830e-001; krok(n+1)=6.128758e-002; ng(n+1)=8.177019e+001;
n=46; farx(n+1)=5.036847e-001; foe(n+1)=7.443133e-001; krok(n+1)=9.363811e-002; ng(n+1)=6.219965e+001;
n=47; farx(n+1)=5.005294e-001; foe(n+1)=7.364743e-001; krok(n+1)=2.580064e-002; ng(n+1)=5.590202e+001;
n=48; farx(n+1)=4.798906e-001; foe(n+1)=7.154011e-001; krok(n+1)=9.714883e-002; ng(n+1)=5.311808e+001;
n=49; farx(n+1)=4.663819e-001; foe(n+1)=7.044398e-001; krok(n+1)=3.782301e-002; ng(n+1)=1.252285e+002;
n=50; farx(n+1)=4.525891e-001; foe(n+1)=6.837914e-001; krok(n+1)=8.877087e-002; ng(n+1)=1.067207e+002;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)