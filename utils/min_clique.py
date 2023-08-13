def prune_graph(G,nodes):
    for n in nodes:
        neighbors = G.pop(n)
        for nn in neighbors:
            G[nn].remove(n)


def degree_ordering(G):
    nodes = list(G.keys())
    return sorted(nodes, reverse=True, key=lambda n: len(G[n]))


def BronKerbosch_pivot(G,R,P,X,cliques):
    """
    For a given graph, G, find a maximal clique containing all of the vertices
    in R, some of the vertices in P, and none of the vertices in X.
    """
    if len(P) == 0 and len(X) == 0:
        # Termination case. If P and X are empty, R is a maximal clique
        cliques.append(R)
    else:
        # choose a pivot vertex
        pivot = next(iter(P.union(X)))
        # Recurse
        for v in P.difference(G[pivot]):
            # Recursion case. 
            BronKerbosch_pivot(G,R.union({v}),P.intersection(G[v]),
                               X.intersection(G[v]),cliques)
            P.remove(v)
            X.add(v)

def BronKerbosch(G):
    """
    Implementation of Bron-Kerbosch algorithm (Bron, Coen; Kerbosch, Joep (1973),
    "Algorithm 457: finding all cliques of an undirected graph", Commun. ACM,
    ACM, 16 (9): 575–577, doi:10.1145/362342.362367.) using a degree ordering
    of the vertices in G instead of a degeneracy ordering.
    See: https://en.wikipedia.org/wiki/Bron-Kerbosch_algorithm
    """

    max_cliques = []

    while len(G) > 0:
        P = set(G.keys())
        R = set()
        X = set()
        v = degree_ordering(G)[0]
        cliques = []
        BronKerbosch_pivot(G,R.union({v}),P.intersection(G[v]),
                           X.intersection(G[v]),cliques)

        #print('i = {}, current v = {}'.format(i,v))
        #print('# cliques: ',len(cliques))

        sorted_cliques = sorted(cliques, key=len, reverse=True)
        max_cliques += [sorted_cliques[0]]
        #print(sorted_cliques[0])

        prune_graph(G,sorted_cliques[0])

    return max_cliques