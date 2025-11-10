######ESTRUTURAS DE CADA TIPO DE CAMADA##########

Base.@kwdef mutable struct conv2d
    k = 0 #tamanho do kernel
    channels = 0 #numero de canais
    stride = 0 #stride
    activation = 0 #funcao de ativacao
    #esses serao inicializados automaticamente:
    pconv = 0 #matriz de apontadores
    H = 0 #kernels
    dH = 0 #derivada dos kernels
    mH = 0
    vH = 0
    B = 0 #bias
    dB = 0 #derivada dos bias
    mB = 0
    vB = 0
    X = 0 #entrada
    Y = 0 #saida
    Ex = 0 #erro entrada
    Ey = 0 #erro saida
    layer = "conv2d"
end

Base.@kwdef mutable struct conv2d_transpose
    k = 0 #tamanho do kernel
    channels_out = 0 #numero de canais
    stride = 0 #stride
    activation = 0 #funcao de ativacao
    #esses serao inicializados automaticamente:
    pconv = 0 #matriz de apontadores
    H = 0 #kernels
    dH = 0 #derivada dos kernels
    mH = 0
    vH = 0
    B = 0 #bias
    dB = 0 #derivada dos bias
    mB = 0
    vB = 0
    X = 0 #entrada
    Y = 0 #saida
    Ex = 0 #erro entrada
    Ey = 0 #erro saida
    layer = "conv2d_transpose"
end

Base.@kwdef mutable struct maxpooling
    k = 0 #tamanho do kernel
    stride = 0 #stride
    #esses serao inicializados automaticamente:
    ppool = 0 #matriz de apontadores
    winners = 0 #matriz que guarda o vencedor de cada caminho
    X = 0 #entrada
    Y = 0 #saida
    Ex = 0 #erro entrada
    Ey = 0 #erro saida
    layer = "pooling"
end

Base.@kwdef mutable struct dense
    n_out = 0 #numero de saídas
    activation = 0 #funcao de ativacao
    #esses serao incializados automaticamente
    W = 0 #pesos
    dW = 0 #derivada dos pesos
    mW = 0
    vW = 0
    X = 0 #entrada
    Y = 0 #saida
    Ex = 0 #erro entrada
    Ey = 0 #erro saida
    layer = "dense"
end

Base.@kwdef mutable struct flatten
    #todas essas são definidas automaticamente
    X = 0 #entrada
    Y = 0 #saida
    Ex = 0 #erro entrada
    Ey = 0 #erro saida
    layer = "flatten"
end

Base.@kwdef mutable struct softmax 
    #todas essas são definidas automaticamente 
    X = 0 #entrada
    Y = 0 #saida
    Ex = 0 #erro entrada
    Ey = 0 #erro saida
    layer = "softmax"
end

######ESTRUTURA DA REDE COMPLETA#############

Base.@kwdef mutable struct network
    nn = nothing #rede neural (coloque aqui a sequencia de camadas como uma lista de structs)
    training = nothing
    t = nothing
end

#######OBTER APONTADORES###########

function obtain_pooling_indexes(M, N, km, kn, stride = 1)
    indexes = Int.(reshape(collect(1:1:M*N), M, N))
    M_out = length(collect(1:stride:M-km+1))
    N_out = length(collect(1:stride:N-kn+1))
    pool_indexes = zeros(Int, km*kn, M_out*N_out)
    cont = 0
    for j in 1:stride:N-kn+1
        for i in 1:stride:M-km+1
            cont = cont+1
            patch = indexes[i:i+km-1, j:j+kn-1]
            pool_indexes[:, cont] = copy(vec(patch))
        end
    end
    return pool_indexes
end



function obtain_conv_indexes(M, N, CH, km, kn, kc, stride = 1)
    indexes = Int.(reshape(collect(1:1:M*N), M, N))
    M_out = length(collect(1:stride:M-km+1))
    N_out = length(collect(1:stride:N-kn+1))
    conv_indexes = zeros(Int, km*kn, M_out*N_out)
    cont = 0
    for j in 1:stride:N-kn+1
        for i in 1:stride:M-km+1
            cont = cont+1
            patch = indexes[i:i+km-1, j:j+kn-1]
            conv_indexes[:, cont] = copy(vec(patch))
        end
    end
    if CH>1
        for ch in 2:CH
            conv_indexes = vcat(conv_indexes, conv_indexes[1:km*kn,:].+(ch-1)*M*N)
        end
    end
    return conv_indexes
end


##############INICIALIZACAO DOS PESOS########################

function init_weights(network, m_in, n_in = 1, c_in = 1)
    #m_in #numero de linhas na entrada
    #n_in #numero de colunas na entrada (somente considerado para primeira camada convolutiva)
    #c_in #numero de canais na entrada (somente considerado para primeira camada convolutiva)

    if network.nn[1].layer == "dense"
        network.nn[1].W = (1/sqrt(m_in + 1))*randn(network.nn[1].n_out, m_in + 1)
        network.nn[1].dW = 0*network.nn[1].W
        network.nn[1].mW = 0*network.nn[1].W
        network.nn[1].vW = 0*network.nn[1].W
        network.nn[1].X = zeros(Float64, m_in)
        network.nn[1].Y = zeros(Float64, network.nn[1].n_out)
        network.nn[1].Ex = zeros(Float64, m_in)
        network.nn[1].Ey = zeros(Float64, network.nn[1].n_out)
        m_in = network.nn[1].n_out
    elseif network.nn[1].layer == "conv2d"
        network.nn[1].pconv = obtain_conv_indexes(m_in, n_in, c_in, network.nn[1].k, network.nn[1].k, network.nn[1].channels, network.nn[1].stride) 
        if c_in == 1
            #network.nn[1].H = (1/sqrt(1+c_in*network.nn[1].k^2))*randn(network.nn[1].k, network.nn[1].k, network.nn[1].channels) #kernels
            network.nn[1].H = (2/sqrt(1+c_in*network.nn[1].k^2))*rand(network.nn[1].k, network.nn[1].k, network.nn[1].channels) .-  1/sqrt(1+c_in*network.nn[1].k^2)#kernels
        else 
            #network.nn[1].H = (1/sqrt(1+c_in*network.nn[1].k^2))*randn(network.nn[1].k, network.nn[1].k, c_in, network.nn[1].channels) #kernels
            network.nn[1].H = (2/sqrt(1+c_in*network.nn[1].k^2))*rand(network.nn[1].k, network.nn[1].k, c_in, network.nn[1].channels) .- 1/sqrt(1+c_in*network.nn[1].k^2)#kernels
        end
        network.nn[1].dH = 0*network.nn[1].H  #derivada dos kernels
        network.nn[1].mH = 0*network.nn[1].H 
        network.nn[1].vH = 0*network.nn[1].H  
        #network.nn[1].B =  (1/sqrt(1+c_in*network.nn[1].k^2))*randn(network.nn[1].channels)#bias
        network.nn[1].B =  (2/sqrt(1+c_in*network.nn[1].k^2))*rand(network.nn[1].channels).- 1/sqrt(1+c_in*network.nn[1].k^2);#bias
        network.nn[1].dB = 0*network.nn[1].B #derivada dos bias
        network.nn[1].mB = 0*network.nn[1].B
        network.nn[1].vB = 0*network.nn[1].B
        if c_in == 1 
            #"cheguei aqui"
            network.nn[1].X = zeros(m_in, n_in)
        else 
            network.nn[1].X = zeros(m_in, n_in, c_in)
        end
        m_out = length(collect(1:network.nn[1].stride:m_in-network.nn[1].k+1))
        n_out = length(collect(1:network.nn[1].stride:n_in-network.nn[1].k+1))
        network.nn[1].Y = zeros(m_out, n_out, network.nn[1].channels) #saida
        network.nn[1].Ex = 0*network.nn[1].X #erro entrada
        network.nn[1].Ey = 0*network.nn[1].Y #erro saida
        m_in = m_out 
        n_in = n_out 
        c_in = network.nn[1].channels 
    end

    for l in 2:length(network.nn)
        if network.nn[l].layer == "dense"
            network.nn[l].W = (1/sqrt(m_in + 1))*randn(network.nn[l].n_out, m_in + 1)
            network.nn[l].dW = 0*network.nn[l].W
            network.nn[l].mW = 0*network.nn[l].W
            network.nn[l].vW = 0*network.nn[l].W
            network.nn[l].X = zeros(Float64, m_in)
            network.nn[l].Y = zeros(Float64, network.nn[l].n_out)
            network.nn[l].Ex = zeros(Float64, m_in)
            network.nn[l].Ey = zeros(Float64, network.nn[l].n_out)
            m_in = network.nn[l].n_out
        elseif network.nn[l].layer == "conv2d"
            network.nn[l].pconv = obtain_conv_indexes(m_in, n_in, c_in, network.nn[l].k, network.nn[l].k, network.nn[l].channels, network.nn[l].stride) 
            if c_in == 1
                #network.nn[l].H = (1/sqrt(1+c_in*network.nn[l].k^2))*randn(network.nn[l].k, network.nn[l].k, network.nn[l].channels) #kernels
                network.nn[l].H = (2/sqrt(1+c_in*network.nn[l].k^2))*rand(network.nn[l].k, network.nn[l].k, network.nn[l].channels) .-  1/sqrt(1+c_in*network.nn[l].k^2)#kernels
            else 
                #network.nn[l].H = (1/sqrt(1+c_in*network.nn[l].k^2))*randn(network.nn[l].k, network.nn[l].k, c_in, network.nn[l].channels) #kernels
                network.nn[l].H = (2/sqrt(1+c_in*network.nn[l].k^2))*rand(network.nn[l].k, network.nn[l].k, c_in, network.nn[l].channels) .- 1/sqrt(1+c_in*network.nn[l].k^2)#kernels
            end
            network.nn[l].dH = 0*network.nn[l].H  #derivada dos kernels
            network.nn[l].mH = 0*network.nn[l].H
            network.nn[l].vH = 0*network.nn[l].H
            #network.nn[l].B =  (1/sqrt(1+c_in*network.nn[l].k^2))*randn(network.nn[l].channels)#bias
            network.nn[l].B =  (2/sqrt(1+c_in*network.nn[l].k^2))*rand(network.nn[l].channels).- 1/sqrt(1+c_in*network.nn[l].k^2);#bias
            network.nn[l].dB = 0*network.nn[l].B #derivada dos bias
            network.nn[l].mB = 0*network.nn[l].B
            network.nn[l].vB = 0*network.nn[l].B
            if c_in == 1 
                network.nn[l].X = zeros(m_in, n_in)
            else 
                network.nn[l].X = zeros(m_in, n_in, c_in)
            end
            m_out = length(collect(1:network.nn[l].stride:m_in-network.nn[l].k+1))
            n_out = length(collect(1:network.nn[l].stride:n_in-network.nn[l].k+1))
            network.nn[l].Y = zeros(m_out, n_out, network.nn[l].channels) #saida
            network.nn[l].Ex = 0*network.nn[l].X #erro entrada
            network.nn[l].Ey = 0*network.nn[l].Y #erro saida
            m_in = m_out 
            n_in = n_out 
            c_in = network.nn[l].channels 
        elseif network.nn[l].layer == "pooling"
            network.nn[l].ppool = obtain_pooling_indexes(m_in, n_in, network.nn[l].k, network.nn[l].k, network.nn[l].stride) #matriz de apontadores
            m_out = length(collect(1:network.nn[l].stride:m_in-network.nn[l].k+1))
            n_out = length(collect(1:network.nn[l].stride:n_in-network.nn[l].k+1))
            network.nn[l].winners = zeros(Int64, 1,m_out*n_out, c_in) #matriz que guarda o vencedor de cada caminho
            network.nn[l].X = zeros(Float64, m_in, n_in, c_in) #entrada
            network.nn[l].Y = zeros(Float64, m_out, n_out, c_in) #saida
            network.nn[l].Ex = 0*network.nn[l].X #erro entrada
            network.nn[l].Ey = 0*network.nn[l].Y #erro saida
            m_in = m_out
            n_in = n_out
        elseif network.nn[l].layer == "flatten"
            network.nn[l].X = zeros(Float64, m_in, n_in, c_in) #entrada
            network.nn[l].Y = copy(vec(network.nn[l].X)) #saida
            network.nn[l].Ex = 0*network.nn[l].X #erro entrada
            network.nn[l].Ey = 0*network.nn[l].Y #erro saida
            m_in = m_in*n_in*c_in
            n_in = 1
            c_in = 1
        elseif network.nn[l].layer == "conv2d_transpose"
            m_out = (m_in-1)*network.nn[l].stride+network.nn[l].k
            n_out = (n_in-1)*network.nn[l].stride+network.nn[l].k
            network.nn[l].pconv = obtain_conv_indexes(m_out, n_out, network.nn[l].channels_out, network.nn[l].k, network.nn[l].k, c_in, network.nn[l].stride)
            network.nn[l].X = zeros(Float64, m_in, n_in, c_in)
            network.nn[l].Y = zeros(Float64, m_out, n_out, network.nn[l].channels_out)
            if network.nn[l].channels_out == 1
                network.nn[l].Y = zeros(Float64, m_out, n_out)
            end
            network.nn[l].Ex = 0*network.nn[l].X
            network.nn[l].Ey = 0*network.nn[l].Y 
            if network.nn[l].channels_out == 1
                #network.nn[l].H = (1/(network.nn[l].k^2+1))*randn(network.nn[l].k,network.nn[l].k,c_in)
                network.nn[l].H = (2/(sqrt(network.nn[l].k^2+1)))*rand(network.nn[l].k,network.nn[l].k,c_in).-1/(sqrt(network.nn[l].k^2+1))
                #network.nn[l].B = (1/(network.nn[l].k^2+1))*randn(m_out, n_out)
                network.nn[l].B = (2/(sqrt(network.nn[l].k^2+1)))*rand(m_out, n_out).-1/(sqrt(network.nn[l].k^2+1))
            else
                #network.nn[l].H = (1/(network.nn[l].k^2+1))*randn(network.nn[l].k,network.nn[l].k,network.nn[l].channels_out, c_in)
                network.nn[l].H = (2/(sqrt(network.nn[l].channels_out*network.nn[l].k^2+1)))*rand(network.nn[l].k,network.nn[l].k,network.nn[l].channels_out, c_in) .- 1/(sqrt(network.nn[l].channels_out*network.nn[l].k^2+1))
                #network.nn[l].B = (1/(network.nn[l].k^2+1))*randn(m_out, n_out, network.nn[l].channels_out)
                network.nn[l].B = (2/(sqrt(network.nn[l].channels_out*network.nn[l].k^2+1)))*rand(m_out, n_out, network.nn[l].channels_out).-1/(sqrt(network.nn[l].channels_out*network.nn[l].k^2+1))
            end
            network.nn[l].dH = 0*network.nn[l].H
            network.nn[l].mH = 0*network.nn[l].H
            network.nn[l].vH = 0*network.nn[l].H
            network.nn[l].dB = 0*network.nn[l].B
            network.nn[l].mB = 0*network.nn[l].B
            network.nn[l].vB = 0*network.nn[l].B
            m_in = m_out
            n_in = n_out
            c_in = network.nn[l].channels_out 
        elseif network.nn[l].layer == "softmax"
            network.nn[l].X = zeros(m_in)
            network.nn[l].Y = zeros(m_in)
            network.nn[l].Ex = zeros(m_in)
            network.nn[l].Ey = zeros(m_in)
        end
    end
    return network
end


#############TESTANDO INICIALIZACAO############
#=
nn_test = model([conv2d(k=3, channels=20, stride=1, activation="relu"),
                 maxpooling(k=2, stride=1),
                 conv2d(k=3, channels=10, stride=1, activation="relu"),
                 maxpooling(k=2, stride=1),
                 flatten(),
                 dense(n_out=50, activation="relu"),
                 dense(n_out=10, activation="relu")                 
])

nn_test = init_weights(nn_test, 16, 16)
=#

#######################FORWARD FUNCTIONS###################

function fc_layer(W::Matrix{Float64}, X::Vector{Float64}, Y::Vector{Float64}, activation::String)
    #Y .= Y*0
    Y .= W*vcat(X, 1)
    if activation == "relu"
        Y .= Y.*(Y.>0)
    elseif activation == "logistic"
        Y .=  1 ./ (exp.(-Y).+1)
    elseif activation == "tanh"
        Y .= tanh.(Y)
    end
end

function conv_layer(H, B::Vector{Float64}, X, Y::Array{Float64,3}, pconv::Matrix{Int64}, activation::String)
    #Y .= Y.*0
    #CALCULANDO DIMENSOES 
    m_in = size(X, 1)
    n_in = size(X,2)
    n_channels_in = size(X,3)
    m_out = size(Y,1)
    n_out = size(Y,2) 
    #=
    if length(size(X))==4
        n_channels_in = size(X,3)
    else
        n_channels_in = 1
    end =#

    n_channels_out = size(H)[end]
    #n_inputs = size(X)[end] 
    ksize = size(H,1)

    vec_H = copy(reshape(H, 1, n_channels_in*ksize^2, n_channels_out))

    #FAZENDO CONVOLUCOES
    for channel in 1:n_channels_out
        Y[:,:,channel] .= reshape(vec_H[:,:,channel]*X[pconv].+B[channel], m_out, n_out)
    end

    #PASSANDO PELA FUNCAO DE ATIVACAO
    if activation == "relu"
        Y .= Y.*(Y.>0)
    elseif activation == "logistic"
        Y .=  1 ./ (exp.(-Y).+1)
    elseif activation == "tanh"
        Y .= tanh.(Y)
    end
end

function max_pooling_layer(X, Y, ppool::Matrix{Int64}, winners_pool::Array{Int64, 3})
    #Y .= Y.*0
    #winners_pool .= winners_pool.*0
    m_out = size(Y,1)
    n_out = size(Y,2)
    for channel in 1:size(X,3)
        Y[:,:,channel] .= reshape([maximum(col) for col in eachcol(X[:,:,channel][ppool])], m_out, n_out)
        aux = [argmax(col) for col in eachcol(X[:,:,channel][ppool])]
        winners_pool[:,:, channel] .= reshape([ppool[aux[w], w] for w in collect(1:1:m_out*n_out)], 1, m_out*n_out)
    end
end

function conv_transpose_layer(X, Y, H, B, pconv, activation)
    #Y .= Y*0
    m_in = size(X,1)
    n_in = size(X,2)
    n_channels_in = size(X,3)
    m_out = size(Y,1)
    n_out = size(Y,2)
    n_channels_out = size(Y,3)
    ksize = size(H,1)
    vec_H = copy(reshape(H, n_channels_out*ksize^2, n_channels_in))
    for i in 1:size(pconv,2)
        for ch in 1:n_channels_in
            Y[pconv[:,i]] .+= (X[:,:,ch][i])*vec_H[:,ch] 
        end
    end
    Y .= Y+B
     #PASSANDO PELA FUNCAO DE ATIVACAO
    if activation == "relu"
        Y .= Y.*(Y.>0)
    elseif activation == "logistic"
        Y .=  1 ./ (exp.(-Y).+1)
    elseif activation == "tanh"
        Y .= tanh.(Y)
    end
end

function flatten(X, Y)
    #Y .= Y.*0
    #Y .= reshape(vec(X), size(X,1)*size(X,2)*size(X,3), size(X,4))
    Y .= copy(vec(X))
end

function softmax_layer(X, Y) 
    Y.= exp.(X)
    Y.= Y/sum(Y)
end 

###############BACKPROPAGATION FUNCTIONS############

function fc_back(X::Vector{Float64}, Y::Vector{Float64}, W::Matrix{Float64}, dW::Matrix{Float64}, Ey::Vector{Float64}, Ex::Vector{Float64}, activation::String)
    #Ex .= Ex*0
    if activation == "relu"
        Ey .= Ey.*(Y.>0)
    elseif activation == "logistic"
        Ey .= Ey.*(Y-Y.^2)
    elseif activation == "tanh"
        Ey .= Ey.*(-Y.^2 .+ 1)
    end
    dW .+= Ey*vcat(X, ones(size(X,2)))'
    Ex .= W[:, 1:end-1]'*Ey
    #if save
    #    push!(ex_history, sum(Ex.^2))
    #end
end

function max_pooling_back(Ey::Array{Float64,3}, Ex::Array{Float64,3}, winners_pool::Array{Int64,3})
    Ex .= Ex.*0
    m_in = size(Ex,1)
    n_in = size(Ex,2)
    m_out = size(Ey,1)
    n_out = size(Ey,2)
    n_channels = size(Ey,3)

    aux = zeros(1, m_in*n_in, n_channels)
    for channel in 1:n_channels
        for w in 1:m_out*n_out
            aux[1, Int(winners_pool[:,:,channel][w]), channel] += (Ey[:,:,channel])[w]  
        end  
    end
    Ex .= copy(reshape(vec(aux), m_in, n_in, n_channels))
end


function conv_back(Ex, Ey, X, Y, H, dH, B, dB, pconv, activation)
    Ex .= Ex*0
    ksize = size(H, 1)
    m_out = size(Ey, 1)
    n_out = size(Ey, 2)
    n_channels_out = size(Ey, 3)
    m_in = size(Ex, 1)
    n_in = size(Ex, 2)
    n_channels_in = size(Ex, 3)

    if activation == "relu"
        Ey .= Ey.*(Y.>0)
    elseif activation == "logistic"
        Ey .= Ey.*(Y-Y.^2)
    elseif activation == "tanh"
        Ey .= Ey.*(-Y.^2 .+ 1)
    end
    
    Eaux = copy(reshape(Ey, 1, m_out*n_out, n_channels_out))
    auxdB = zeros(length(dB))

    if n_channels_in>1
        for channel in 1:n_channels_out
            dH[:,:,:,channel] .= dH[:,:,:,channel] + reshape(Eaux[:,:,channel]*X[pconv'], ksize, ksize, n_channels_in)
            auxdB[channel] = auxdB[channel] + sum(Eaux[:,:,channel])
            #dB[channel] .= dB[channel] + sum(Eaux[:,:,channel])
        end
        dB .= dB + auxdB 

        #aux = zeros(1, m_in*n_in*n_channels_in)
        vec_H = copy(reshape(H, 1, n_channels_in*ksize^2, n_channels_out))
        for channel in 1:n_channels_out
            for back in 1:size(Eaux, 2)
                #aux[1, pconv[:,back]] += vec(vec_H[:,:,channel]*Eaux[1, back, channel, input])
                Ex[pconv[:,back]] .+= vec(vec_H[:,:,channel]*Eaux[1,back,channel])
            end
        end

        #Ex .= reshape(aux, m_in, n_in, n_channels_in, n_inputs)
    else 
        for channel in 1:n_channels_out
            dH[:,:,channel] .= dH[:,:,channel] + reshape(Eaux[:,:,channel]*X[pconv'], ksize, ksize)
            auxdB[channel] = auxdB[channel] + sum(Eaux[:,:,channel])
        end
        dB .= dB + auxdB 
    end
end

function conv_transpose_back(Ex, Ey, X, Y, H, dH, dB, pconv, activation)
    ksize = size(H, 1)
    m_out = size(Ey, 1)
    n_out = size(Ey, 2)
    n_channels_out = size(Ey, 3)
    m_in = size(Ex, 1)
    n_in = size(Ex, 2)
    n_channels_in = size(Ex, 3)

    if activation == "relu"
        Ey .= Ey.*(Y.>0)
    elseif activation == "logistic"
        Ey .= Ey.*(Y-Y.^2)
    elseif activation == "tanh"
        Ey .= Ey.*(-Y.^2 .+ 1)
    end

    dB .= copy(Ey)
    Ex_aux = zeros(size(Ex))
    #println(sum(Ey.^2))
    for ch in 1:n_channels_in
        for i in 1:size(pconv,2)
            if n_channels_out > 1
                dH[:,:, :, ch] .+= reshape((X[:,:,ch][i])*Ey[pconv[:,i]], ksize, ksize, n_channels_out)
                var_aux = @view(Ex_aux[:,:,ch])
                var_aux[i] = sum(Ey[pconv[:,i]].*vec(H[:,:,:,ch]))
                #Ex_aux[:,:,ch] = copy(var_aux)
                #Ex_aux[:,:,ch][i] = sum(Ey[pconv[:,i]].*vec(H[:,:,:,ch]))
            else 
                dH[:,:, ch] .+= reshape((X[:,:,ch][i])*Ey[pconv[:,i]], ksize, ksize)
                var_aux = @view(Ex_aux[:,:,ch])
                var_aux[i] = sum(Ey[pconv[:,i]].*vec(H[:,:,ch]))
                #Ex_aux[:,:,ch] = copy(var_aux)
                #Ex_aux[:,:,ch][i] = sum(Ey[pconv[:,i]].*vec(H[:,:,ch]))
                 #if i == 1
                #    println(sum(Ex_aux))
                #end
            end
        end
    end
    Ex .= copy(Ex_aux)
    #println(sum(Ex.^2))
end


function softmax_back(Ex, Ey)
    Ex .= copy(Ey)
end


############IDA E VOLTA DA REDE#############


function forward_nn(X0)
    #println(size(X0))
    #println(size(model.nn[1].X))
    model.nn[1].X .= copy(X0)
    for l in 1:length(model.nn)
        if model.nn[l].layer == "dense"
            fc_layer(model.nn[l].W, model.nn[l].X, model.nn[l].Y, model.nn[l].activation)
        elseif model.nn[l].layer == "conv2d"
            conv_layer(model.nn[l].H, model.nn[l].B, model.nn[l].X, model.nn[l].Y, model.nn[l].pconv, model.nn[l].activation)
        elseif model.nn[l].layer == "pooling"
            max_pooling_layer(model.nn[l].X, model.nn[l].Y, model.nn[l].ppool, model.nn[l].winners)
        elseif model.nn[l].layer == "flatten"
            flatten(model.nn[l].X, model.nn[l].Y)
        elseif model.nn[l].layer == "conv2d_transpose"
            conv_transpose_layer(model.nn[l].X, model.nn[l].Y, model.nn[l].H, model.nn[l].B, model.nn[l].pconv, model.nn[l].activation)
        elseif model.nn[l].layer == "softmax"
            softmax_layer(model.nn[l].X,model.nn[l].Y)
        end
        
        if l !=  length(model.nn)
            #println(l)
            model.nn[l+1].X .= copy(model.nn[l].Y)
            if model.training == 0
                model.nn[l].Y .= model.nn[l].Y*0
            end
        end
    end 
end

#ex_history = []

function backpropagation_nn(output)
    model.nn[end].Ey .= model.nn[end].Y - output
    for l in length(model.nn):-1:1
        if model.nn[l].layer == "dense"
            fc_back(model.nn[l].X, model.nn[l].Y, model.nn[l].W, model.nn[l].dW, model.nn[l].Ey, model.nn[l].Ex, model.nn[l].activation)
        elseif model.nn[l].layer == "conv2d"
            conv_back(model.nn[l].Ex, model.nn[l].Ey, model.nn[l].X, model.nn[l].Y, model.nn[l].H, model.nn[l].dH, model.nn[l].B, model.nn[l].dB, model.nn[l].pconv, model.nn[l].activation)
        elseif model.nn[l].layer == "pooling"
            max_pooling_back(model.nn[l].Ey, model.nn[l].Ex, model.nn[l].winners)
        elseif model.nn[l].layer == "flatten"
            model.nn[l].Ex .= copy(reshape(model.nn[l].Ey, size(model.nn[l].Ex,1), size(model.nn[l].Ex, 2), size(model.nn[l].Ex,3)))
        elseif model.nn[l].layer == "conv2d_transpose"
            conv_transpose_back(model.nn[l].Ex, model.nn[l].Ey, model.nn[l].X, model.nn[l].Y, model.nn[l].H, model.nn[l].dH, model.nn[l].dB, model.nn[l].pconv, model.nn[l].activation)
        elseif model.nn[l].layer == "softmax"
            softmax_back(model.nn[l].Ex, model.nn[l].Ey)
        end 

        
        
        if l != 1
            model.nn[l-1].Ey .= copy(model.nn[l].Ex)
        end
        model.nn[l].Ex .= model.nn[l].Ex*0
        model.nn[l].Ey .= model.nn[l].Ey*0
        model.nn[l].Y .= model.nn[l].Y*0
    end
end

function update_model(optimizer)
    global lr
    global beta1
    global beta2
    global epsilon
    #global cont
    model.t += 1
    for l in 1:length(model.nn)
        if model.nn[l].layer == "dense"
            if optimizer == "sgd"
                model.nn[l].W .-= lr*model.nn[l].dW
            elseif optimizer ==  "adam"
            #if (cont == 2 && l ==4)
            #    println(sum((model.nn[l].dW).^2))
            #end          
                model.nn[l].mW .= copy(model.nn[l].mW*beta1 + (1-beta1)*model.nn[l].dW)
                model.nn[l].vW .= copy(model.nn[l].vW*beta2 + (1-beta2)*(model.nn[l].dW).^2)
                aux = model.nn[l].vW/(1-beta2^model.t)
                model.nn[l].W .-= lr*((model.nn[l].mW/(1-beta1^model.t))./(sqrt.(aux).+epsilon)) 
            end
            #if l == 4
            #    push!(ex_history, sum((model.nn[l].dW).^2))
            #end
             #if (model.t == 337) && (l==5)
             #   println((model.nn[l].W))
            #end 
            #resetando os gradientes
            model.nn[l].dW .= 0*model.nn[l].dW
        elseif model.nn[l].layer == "conv2d"
            if optimizer == "sgd"
                model.nn[l].H .-= lr*model.nn[l].dH 
                model.nn[l].B .-= lr*model.nn[l].dB 
            elseif optimizer == "adam"
                model.nn[l].mH .= copy(model.nn[l].mH*beta1 + (1-beta1)*model.nn[l].dH)
                model.nn[l].vH .= copy(model.nn[l].vH*beta2 + (1-beta2)*(model.nn[l].dH).^2)
                model.nn[l].H .-= lr*(model.nn[l].mH/(1-beta1^model.t))./(sqrt.(model.nn[l].vH/(1-beta2^model.t)).+epsilon)
                model.nn[l].mB .= copy(model.nn[l].mB*beta1 + (1-beta1)*model.nn[l].dB)
                model.nn[l].vB .= copy(model.nn[l].vB*beta2 + (1-beta2)*(model.nn[l].dB).^2)
                model.nn[l].B .-= lr*(model.nn[l].mB/(1-beta1^model.t))./(sqrt.(model.nn[l].vB/(1-beta2^model.t)).+epsilon)
            end
           #resetando os gradientes 
            model.nn[l].dH .= 0*model.nn[l].dH
            model.nn[l].dB .= 0*model.nn[l].dB
        elseif model.nn[l].layer == "conv2d_transpose"
            if optimizer == "sgd"
                model.nn[l].H .-= lr*model.nn[l].dH 
                model.nn[l].B .-= lr*model.nn[l].dB 
            elseif optimizer == "adam"
                model.nn[l].mH .= model.nn[l].mH*beta1 + (1-beta1)*model.nn[l].dH 
                model.nn[l].vH .= model.nn[l].vH*beta2 + (1-beta2)*model.nn[l].dH.^2 
                model.nn[l].H .-= lr*(model.nn[l].mH/(1-beta1^model.t))./(sqrt.(model.nn[l].vH/(1-beta2^model.t)).+epsilon)
                model.nn[l].mB .= model.nn[l].mB*beta1 + (1-beta1)*model.nn[l].dB 
                model.nn[l].vB .= model.nn[l].vB*beta2 + (1-beta2)*model.nn[l].dB.^2 
                model.nn[l].B .-= lr*(model.nn[l].mB/(1-beta1^model.t))./(sqrt.(model.nn[l].vB/(1-beta2^model.t)).+epsilon)
            end 
            #resetando os gradientes 
            model.nn[l].dH .= 0*model.nn[l].dH
            model.nn[l].dB .= 0*model.nn[l].dB
        end 
    end
end
function train_nn(batches, X_im, outputs, optimizer = "sgd", more_than_one_input = true)
    model.training = 1
    order_batches = sortperm(rand(length(batches)))
    dim_in = length(size(X_im))
    for batch in order_batches
        for i in 1:length(batches[batch])
            if dim_in == 2
                X0 = copy(X_im[:, batches[batch][i]])
            elseif dim_in == 3
                X0 = copy(X_im[:, :, batches[batch][i]])
            elseif dim_in == 4
                X0 = copy(X_im[:, :, :, batches[batch][i]])
            end
            idx = ntuple(_ -> Colon(), length(size(outputs))-1)
            output = outputs[idx...,batches[batch][i]]
            if !more_than_one_input
                output = copy(outputs)
            end
            forward_nn(X0)
            backpropagation_nn(output)
        end
        update_model(optimizer)
    end
end


function evaluate_nn(inputs, ground_truth, more_than_one_input = true)
    #println("cheguei aqui")
    model.training = 0
    n_inputs = size(inputs)[end]
    if !more_than_one_input
        n_inputs = 1
    end
    err = 0
    c_corrects = 0
    dim_in = length(size(inputs))
    #println(dim_in)
    for i in 1:n_inputs
        if dim_in == 2
            X0 = inputs[:,i]
        elseif dim_in == 3
            #println("cheguei")
            X0 = inputs[:,:,i]
        elseif dim_in == 4
            X0 = inputs[:,:,:,i]
        end

        if n_inputs == 1
            X0 = copy(inputs)
        end
        idx = ntuple(_ -> Colon(), length(size(ground_truth))-1)
        output = ground_truth[idx..., i]
        if !more_than_one_input
            output = copy(ground_truth)
        end

        forward_nn(X0)
        err += mean((model.nn[end].Y - output).^2)/n_inputs
        c_corrects += (argmax(model.nn[end].Y) == argmax(output))
        model.nn[end].Y .= model.nn[end].Y*0
    end
    acc = c_corrects/n_inputs 
    return err, acc 
end 


#evaluate_nn(inputs, ground_truth)


#####DEFININDO MODELO######
#=
model = network([conv2d(k=5, channels=6, stride=1, activation="tanh"),
                 maxpooling(k=2, stride=2),
                 conv2d(k=3, channels=16, stride=1, activation="tanh"),
                 maxpooling(k=2, stride=2),
                 flatten(),
                 dense(n_out=120, activation="tanh"),
                 dense(n_out=84, activation="tanh"),
                 dense(n_out=10, activation="tanh"),
                 softmax() 
], 0, 0)


model = init_weights(model, 28, 28) =# #inicializando parametros da rede