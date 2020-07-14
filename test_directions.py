import Geometric.Directions.DNA_directions_convex as dna_directions

dna = ((-1, 1, 3, 32, 32), (0, 3, 4, 3, 3), (0, 4, 5, 3, 3, 2), (0, 5, 5, 3, 3), (0, 5, 6, 3, 3, 2), (0, 6, 7, 3, 3), (0, 7, 8, 8, 8), (1, 8, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 5, 6), (3, 6, 7))

new_dna = dna_directions.add_layer(3,dna)

print("new dna: ", new_dna)