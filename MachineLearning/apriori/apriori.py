from numpy import *

def load_data_set():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_C1(data_set):
    """
    :param data_set:
    :return:
        C1_frozenset: use frozen set so we can use it as a key in a dict
    """
    C1 = []
    for transaction in data_set:
        for item in transaction:
            if not [item] in C1:  #store all the item unrepeatly
                C1.append([item])

    C1.sort()

    return list(map(frozenset, C1))



def scan_D(D,Ck,min_support):
    """

    :param D:
    :param Ck:
    :param min_support:
    :return:
    """
    ss_cnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ss_cnt:
                    ss_cnt[can] = 1
                else:
                    ss_cnt[can] += 1
    num_items = float(len(D))
    retList = []
    support_data = {}
    for key in ss_cnt:
        support = ss_cnt[key] / num_items
        if support >= min_support:
            retList.insert(0,key)
        support_data[key] = support
    return retList, support_data



def apriori_gen(Lk, n):  # creates Ck
    """
    using L_k generate L_{k+1}
    :param Lk: Lk
    :param n: n = k + 1
    :return: Cn
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            La = list(Lk[i])[:n - 1]; Lb = list(Lk[j])[:n - 1]
            La.sort(); Lb.sort()
            if La == Lb:  # if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j])  # set union
    return retList



def apriori(data_set, min_support=0.5):
    C1 = create_C1(data_set)
    D = list(map(set, data_set))
    L1, support_data = scan_D(D, C1, min_support)
    L = [L1]
    k = 1
    while (len(L[k - 1]) > 0): # if L[] is not empty
        Ck = apriori_gen(L[k - 1], k) # using L[k-1] gen C[k]
        Lk, supK = scan_D(D, Ck, min_support)  # scan DB to get Lk
        support_data.update(supK)
        L.append(Lk)
        k += 1
    return L, support_data



def generate_rules(L, support_data, min_conf=0.7):
    """
    Association rule generation, {p => H}
    :param L:
    :param support_data:
    :param min_conf:
    :return:
    """
    big_rule_list = []    # store all association rules
    for i in range(1, len(L)):  # Get only sets with two or more items
        for freq_set in L[i]:
            # Loop each freq items of L and
            H1 = [frozenset([item]) for item in freq_set]
            calc_conf(freq_set, H1, support_data, big_rule_list, min_conf)
            if (i > 1):
                rules_from_conseq(freq_set, H1, support_data, big_rule_list, min_conf)
            # else:
            #     calc_conf(freq_set, H1, support_data, big_rule_list, min_conf)
    return big_rule_list



def calc_conf(freq_set, H, support_data, brl, min_conf=0.7):
    """
    Calculate the confidence of a rule and then find out which rules meet the minimum confidence
    confidence(P => H) = support(P & H) / support(P)
    lift(P H) = support(P & H) / (support(P)*support(H))
    KULC(P H) = (p(P|H) + p(H|P)) / 2
    IR(P => H) = p(H|P) / p(P|H))
    :param freq_set:
    :param H:
    :param support_data:
    :param brl:
    :param min_conf:
    :return: prunedH: a list of rules that meet the min- imum confidence
    """
    prunedH = []
    for conseq in H:
        conf = support_data[freq_set] / support_data[freq_set-conseq]
        lift = support_data[freq_set] / (support_data[conseq] * support_data[freq_set-conseq])
        kulc = (support_data[freq_set] / support_data[freq_set-conseq] + support_data[freq_set] / support_data[conseq]) / 2
        ir = support_data[conseq] / support_data[freq_set-conseq]
        print(freq_set - conseq, "\t-->\t", conseq,
              "\tconf:", round(conf, 4),
              "\tlift:", round(lift, 4),
              "\tkulc:", round(kulc, 4),
              "\tir:", round(ir, 4))
        if conf >= min_conf:
            # print (freq_set-conseq, "\t-->\t", conseq,
            #        "\tconf:", round(conf, 4),
            #        "\tlift:", round(lift, 4),
            #        "\tkulc:", round(kulc, 4),
            #        "\tir:", round(ir, 4))
            brl.append((freq_set-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH



def rules_from_conseq(freq_set, H, support_data, brl, min_conf=0.7):
    """

    :param freq_set:
    :param H:
    :param support_data:
    :param brl:
    :param min_conf:
    :return:
    """
    m = len(H[0])
    # if (len(freq_set) > (m + 1)):
    if (len(freq_set) > m + 1):
        Hmp1 = apriori_gen(H, m)    # Create Hm+1 new candidates
        Hmp1 = calc_conf(freq_set, Hmp1, support_data, brl, min_conf)
        if (len(Hmp1) > 1):
            rules_from_conseq(freq_set, Hmp1, support_data, brl, min_conf)
