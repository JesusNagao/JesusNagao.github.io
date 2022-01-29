using Plots;
include("Operations.jl");


print(Threads.nthreads())

x = Array{Float64}([i for i in range(0.0, stop=1000.0, step=10)]);
y = Array{Float64}(undef, length(x));
u = Array{Float64}(zeros(length(x)));
h = Array{Float64}(undef, length(x));

u[1] = 0;
u[length(x)] = 0;

const dx = 10.0;
const g = 9.8;
const h0 = 10.0;
it = 600;

y = gaussian(x, y, 250.0, 250.0, 750.0);
h = height(h, y, h0);
h_max = maxim(h);

dt = dx*0.8/sqrt(abs(h_max*g));

@time animate(x, y, u, h, it, g, dx, dt, 0.05, h0);
