import apriori

if __name__ == '__main__':
    dataSet = apriori.loadDataSet()
    print "dataset:", dataSet
    C1 = apriori.createC1(dataSet)
    D = map(set, dataSet)
    print "D:", D
    print "C1:", C1
    L1,suppData0 = apriori.scanD(D, C1, 0.5)
    print "L1:", L1
    print "suppData0:", suppData0

# dataset: [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
# D: [set([1, 3, 4]), set([2, 3, 5]), set([1, 2, 3, 5]), set([2, 5])]
# C1: [frozenset([1]), frozenset([2]), frozenset([3]), frozenset([4]), frozenset([5])]
# L1: [frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]
# suppData0: {frozenset([4]): 0.25, frozenset([5]): 0.75, frozenset([2]): 0.75, frozenset([3]): 0.75, frozenset([1]): 0.5}
