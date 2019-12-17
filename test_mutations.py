from mutations.Dictionary import MutationsDictionary

redVieja = "redVieja"
adnViejo = ((0, 3, 100, 2, 3), (1, 100, 2), (2,))
adnNuevo = ((0, 3, 101, 2, 3), (1, 101, 2), (2,))

dictionaryMutation = MutationsDictionary()


for i in range(2):
    mutation = dictionaryMutation.getMutation(adnViejo[i], adnNuevo[i])
    mutation.doMutate(redVieja, adnNuevo)