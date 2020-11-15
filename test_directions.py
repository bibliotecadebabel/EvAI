
import Geometric.Directions.DNA_directions_pool_duplicate as dna_directions
import Geometric.Directions.DNA_directions_convex as dna_directions_convex

dna =  ((-1, 1, 3, 32, 32), (0, 3, 5, 3, 3), (0, 5, 10, 4, 4), (0, 10, 15, 2, 2), 
                                (0, 15, 30, 26, 26), 
                                (1, 30, 10), 
                                (2,), 
                                (3, -1, 0), 
                                (3, 0, 1), 
                                (3, 1, 2), 
                                (3, 2, 3), 
                                (3, 3, 4), 
                                (3, 4, 5))

dna_convex = ((-1, 1, 3, 32, 32), (0, 3, 5, 3, 3), (0, 5, 10, 4, 4), (0, 10, 15, 2, 2), 
                                (0, 15, 30, 5, 5), 
                                (1, 30, 10), 
                                (2,), 
                                (3, -1, 0), 
                                (3, 0, 1),
                                (3, 1, 2), 
                                (3, 2, 3), 
                                (3, 3, 4), 
                                (3, 4, 5))
                   
new_dna = dna_directions.add_layer(2,dna)
new_dna_convex = dna_directions_convex.add_layer(2, dna_convex)


print("new dna: ", new_dna)
#print(" ")
#print("new convex: ", new_dna_convex)
