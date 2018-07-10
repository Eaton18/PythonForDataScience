
import python.machinelearning.apriori.apriori as apriori

# Test association rules
if False:
    data_set = apriori.load_data_set()
    L, support_data = apriori.apriori(data_set)
    print("Support_data: ", support_data)
    print("L[0]: ", L[0])
    print("L[1]: ", L[1])
    print("L[2]: ", L[2])
    print("L[3]: ", L[3])
    print("*******************************")

    rules = apriori.generate_rules(L, support_data, min_conf=0.4)
    print(rules)


# Test creat C1, L1
if True:
    data_set = apriori.load_data_set()
    print("data_set:\n", data_set)

    C1 = apriori.create_C1(data_set)
    print("C1:\n", C1)

    D = list(map(set, data_set))
    print("D:\n", D)
    # print(list(map(set, data_set)))

    L1, supp_data0 = apriori.scan_D(D, C1, 0.5)
    print("L1:\n", L1)
    print("Supp_data0:\n", supp_data0)


# Test gen frequency items Lk
if False:
    data_set = apriori.load_data_set()
    L, support_data = apriori.apriori(data_set)

    print("L[0]: ", L[0])
    print("L[1]: ", L[1])
    print("L[2]: ", L[2])
    print("L[3]: ", L[3])

    print(">> Apriori Gen:")
    print(apriori.apriori_gen(L[0], 1))

    L, support_data = apriori.apriori(data_set, min_support=0.7)
