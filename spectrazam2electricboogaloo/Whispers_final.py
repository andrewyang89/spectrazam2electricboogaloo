#!/usr/bin/env python
# coding: utf-8

# In[1]:


def whispers(descriptors, file_paths, iterations, threshold):       
    """ 
        Parameters
        ----------

        descriptors : numpy.ndarray(N, 512)
            The descriptor vector for each N pictures

        iterations : int
            The number of times the Whispers algorithm should iterate

        threshold : int
            The value at which to determine if two pictures are the same or not
            
            
        """
    
    N = descriptors.shape[0]
    
    adj = np.zeros((N,N))

    cos_dis = np.zeros((N,N))
    
    
    #Compute Cosine Values between all descriptor vectors
    for i in range(N):
        for j in range(N):
            if i != j:
                cos_dis[i][j] = cos_dist(descriptors[i, :], descriptors[j, :])
    
    
    
    #Compute Adj matrix
    for i in range(N):
        for j in range(N):
            adj[i][j] = 0
            if cos_dis[i][j] < threshold and i != j:
                adj[i][j] = 1 / (cos_dis[i][j] ** 2)
    
    

    
    
    #Create all nodes along with respective neighbors
    all_nodes = np.ndarray((N,), dtype = Node )

    for i in range(N):
        neighbors = []
        ID = i
        for j in range(N):
            if adj[i][j] != 0:
                neighbors.append(j)

        all_nodes[i] = Node(ID, neighbors, descriptors[i, :], file_path = file_paths[i])
        
    
    
    
    
    #Go through actual iterations of the Whispers Algorithm 
    for i in range(iterations):
        current_node_idx = np.random.randint(N,)

        c = current_node_idx
        dictionary = {}

        for j in range(N):


            if all_nodes[j].label in dictionary.keys():
                dictionary[all_nodes[j].label] = dictionary[all_nodes[j].label] + adj[c][j]

            else:
                dictionary[all_nodes[j].label] = adj[c][j]



        new_label = 0
        max_weight = 0

        for idx,val in dictionary.items():
            if val > max_weight:
                max_weight = val
                new_label = idx

        #print(new_label)
        all_nodes[c].label = new_label
    
    
    
    #print total number of unique people
    total = set()
    for i in range(N):
       # print(all_nodes[i].label)
        total.add(all_nodes[i].label)


    
    
    #Plot graph
    
    
    print("total number of people:", len(total))  
    
    plot_graph(tuple(all_nodes), adj)
    
    sorted_nodes = sorted(all_nodes, key = lambda Node: Node.label)

    prev = sorted_nodes[0].label
    num = 1
    count = 0
    for i in sorted_nodes:
        
        if(i.label != prev):
            print("Person:", num, "   Number of photos", count)
            num+=1
            count = 1
        else:
            count+=1
        
        pic = plt.imread(i.file_path)
        fig, ax = plt.subplots()
        ax.imshow(pic)
        
        prev = i.label
        
    print("Person:", num, "   Number of photos", count)
        

    


# In[ ]:




