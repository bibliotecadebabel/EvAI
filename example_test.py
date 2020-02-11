
## MUTACION DE PROFUNDIDAD DE KERNEL
def Test_DepthKernel(dataGen):
    batch = [dataGen.data]
    print("len data: ", len(dataGen.data[0]))
    ks = [10]
    x = dataGen.size[1]
    y = dataGen.size[2]

    print("creating networks")
    #(0, ks[i], len(dataGen.data[0]), 1, 1),

    
    networkADN = ((0, 3, 9, 1, 2, 2), (0, 1, 5, 3, 10, 10), (0, 7, 20, 5, 1, 1), (1, 20, 10), (1, 10, 2), (2,))
    mutationADN = ((0, 3, 9, 1, 2, 2), (0, 1, 5, 2, 10, 10), (0, 8, 20, 5, 1, 1), (1, 20, 10), (1, 10, 2), (2,))
    network = nw.Network(networkADN, cudaFlag=True)

    for _,a in enumerate(batch):
        print("red original (network) k=", *network.adn)
        network.Training(data=a[0], p=200, dt=0.01, labels=a[1])
        print("mutando")
        network2 = MutateNetwork.executeMutation(network, mutationADN)
        print("entrando red mutada (network2): ", *network2.adn)
        network2.Training(data=a[0],p=200, dt=0.01, labels=a[1])
        print("entrenando de nuevo red original (network)")
        network.Training(data=a[0], p=200, dt=0.01, labels=a[1])
        #Inter.trakPytorch(network, "pokemon-netmap", dataGen)
