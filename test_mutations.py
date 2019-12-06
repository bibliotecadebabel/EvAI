from mutations.Dictionary import MutationsDictionary

redVieja = "redVieja"
adnNuevo = ((0,1,2,3), (1,2,3,4))

dictionaryMutation = MutationsDictionary()
operations = [-1, 1]

for i in range(2):
    for operation in operations:
        mutation = dictionaryMutation.getMutation(i, operation)
        mutation.doMutate(redVieja, adnNuevo)