syms phi tx ty px py sx sy nx ny
R = [1 -phi; phi, 1];
p = [px; py];
s = [sx; sy];

t = [tx; ty];
n = [nx; ny];

M = simple(transpose(t - R * s - p) * n);

b = -simple(subs(M, [phi, px, py], [0, 0, 0])) % (tT - sT) * n

x = [phi; px; py]

aphi = simple(subs(M - b, transpose(x), [1, 0, 0]));
apx = simple(subs(M - b, transpose(x), [0, 1, 0]));
apy = simple(subs(M - b, transpose(x), [0, 0, 1]));


A = [aphi, apx, apy]