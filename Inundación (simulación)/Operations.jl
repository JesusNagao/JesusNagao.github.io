using Plots;
using Base.Threads;


function gaussian(x::Array{Float64}, y::Array{Float64}, mu::Float64, s::Float64, mu2::Float64)
    
    for i in range(1, stop = length(y))
        y[i] = exp(-((x[i]-mu)^2)/s)+exp(-((x[i]-mu2)^2)/s);
    end

    return y;

end

function height(h::Array{Float64}, y::Array{Float64}, h0::Float64)

    for i in range(1, stop=length(h))

        h[i] = y[i] + h0;


    end

    return h
end

function maxim(x::Array{Float64})
    x_max = x[1];

    for i in range(1, stop=length(x))

        if (x[i] > x_max)
            x_max = x[i];
        end

    end
    
    return x_max;

end

function animate(x::Array{Float64}, y::Array{Float64}, u::Array{Float64}, h::Array{Float64}, it::Int64, g::Float64, dx::Float64, dt::Float64, eps::Float64, h0::Float64)


    @gif for j in range(1, stop = it)

        p1 = scatter(x,y, title="Position")
        ylims!(-1.0, 1.0)
        xlims!(0.0, 1000.0)
        p2 = scatter(x,u, title="Velocity")
        ylims!(-1.0, 1.0)
        xlims!(0.0, 1000.0)
        p3 = scatter(x, h, title="Height")
        ylims!(0.0, 20.0)
        xlims!(0.0, 1000.0)
        plot(p1, p2, p3, layout=(3,1), legend=false)


        @threads for i in range(3, stop=length(y)-2)
            
            if i>49 && i<51
                u[i] = 0;
                up = 0.5*(u[i]+abs(u[i]));
                um = 0.5*(u[i]-abs(u[i]));
                up_ant = 0.5*(u[i-1]+abs(u[i-1]));
                um_ant = 0.5*(u[i-1]-abs(u[i-1]));

                np = y[i] - (dt/dx)*(up*h[i]+um*h[i+1]-up_ant*h[i-1]-um_ant*h[i]);
                np_next = y[i+1] - (dt/dx)*(up*h[i+1]+um*h[i+2]-up_ant*h[i]-um_ant*h[i+1]);
                np_ant = y[i-1] - (dt/dx)*(up*h[i-1]+um*h[i]-up_ant*h[i-2]-um_ant*h[i-1]);

                y[i] = 0;

            else
                u[i] = u[i]-(g*dt/dx)*(y[i+1]-y[i]);

                up = 0.5*(u[i]+abs(u[i]));
                um = 0.5*(u[i]-abs(u[i]));
                up_ant = 0.5*(u[i-1]+abs(u[i-1]));
                um_ant = 0.5*(u[i-1]-abs(u[i-1]));

                np = y[i] - (dt/dx)*(up*h[i]+um*h[i+1]-up_ant*h[i-1]-um_ant*h[i]);
                np_next = y[i+1] - (dt/dx)*(up*h[i+1]+um*h[i+2]-up_ant*h[i]-um_ant*h[i+1]);
                np_ant = y[i-1] - (dt/dx)*(up*h[i-1]+um*h[i]-up_ant*h[i-2]-um_ant*h[i-1]);

                y[i] = (1-eps)*np+0.5*eps*(np_ant+np_next);
            end
        end

        h = height(h, y, h0)
        
        

    end

end