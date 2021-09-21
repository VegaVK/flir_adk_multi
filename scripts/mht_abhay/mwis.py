import numpy as np
import operator
import random


def main():
    G = graph()
 

class graph():

    def __init__(self):
        pass
        
    def mwis(self,g):
        complement = g
        # complement = np.array([[0,0,0,0,1,1,1,1],[0,0,0,0,1,1,1,1],[0,0,0,0,0,0,1,1],[0,0,0,0,0,0,1,1],[1,1,0,0,0,0,0,1],[1,1,0,0,0,0,0,1],[1,1,1,1,0,0,0,1],[1,1,1,1,1,1,1,0]])
        ind_sets = []
        call = []
        self.bron_kerbosch3(complement, ind_sets, call)
        # call.append(1)
        # print(ind_sets)
        # print(len(call))

        return ind_sets




    def bron_kerbosch3(self, g, results,count):
        """With vertex ordering."""
        n = len(g)
        # n = 8
        P = set(range(n))
        R, X = set(), set()
        deg_ord = self.__degeneracy_ordering(g)
        # print(deg_ord)
       

        for v in deg_ord:
            # print('v3',v)
            # raw_input('Press ENTER to continue...')
            N_v = self.__n(v, g)
            # print(P,X,N_v)
            # print(R)
            # print(R | {v}, P & N_v, X & N_v)
            self.bron_kerbosch2(R | {v}, P & N_v, X & N_v, g, results,count)
            count.append(1)

            P = P - {v}
            X = X | {v}

        # return results

    def bron_kerbosch2(self, R, P, X, g, results,count):
        """With pivoting."""
        # print('in bk2 RPX',R,P,X)
        if not any((P, X)):
            # print('adding result2',R)
            results.append(R)
            return

        u = random.choice(tuple(P | X))
        # print('u',u)
        # print('range of v2',P - self.__n(u, g))
        for v in P - self.__n(u, g):
            # print('v2',v)
            N_v = self.__n(v, g)
            # print('N_v',N_v)
            # print(R | {v}, P & N_v, X & N_v)
            # raw_input('Press ENTER to continue...')
            self.bron_kerbosch2(R | {v}, P & N_v, X & N_v, g, results,count)
            # count.append(1)

            P = P - {v}
            X = X | {v}

    # def bron_kerbosch(self, R, P, X, g, results):
    #     """Without pivoting."""
    #     print('in bk1 RPX',R,P,X)
    #     if not any((P, X)):
    #         results.append(R)
    #         print('adding result',R)

    #     for v in set(P):
    #         print('v',v)
    #         N_v = self.__n(v, g)
    #         print('N_v',N_v)
    #         print(R | {v}, P & N_v, X & N_v)
    #         raw_input('Press ENTER to continue...')
    #         self.bron_kerbosch(R | {v}, P & N_v, X & N_v, g, results)

    #         P = P - {v}
    #         X = X | {v}


    def __n(self, v, g):
        return set([i for i, n_v in enumerate(g[v]) if n_v])


    def __degeneracy_ordering(self, g):
        """Order such that each vertex has d or fewer neighbors that come later in the ordering."""
        v_ordered = []
        degrees = list(enumerate(self.vertex_degrees(g)))
        # print(degrees)
        while degrees:
            # min_index, min_value = min(degrees, key=operator.itemgetter(1))
            # v_ordered.append(min_index)
            # degrees.remove((min_index, min_value))
            max_index, max_value = max(degrees, key=operator.itemgetter(1))
            v_ordered.append(max_index)
            # print(max_index)
            degrees.remove((max_index, max_value))

        return v_ordered

    def vertex_degrees(self, adj_mat):
        
        degrees = adj_mat.sum(axis=1)
        return degrees

        
if __name__=='__main__':

	main()