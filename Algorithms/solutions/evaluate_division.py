
"""
Leetcode 399 Evaluate Division
https://leetcode.com/problems/evaluate-division/description/
"""

# equations = [ ["a","b"],["b","c"] ]
# values = [2.0,3.0]
# queries = [ ["a","c"],["b","c"],["a","e"],["a","a"],["x","x"] ]

import collections

class EvaluateDivision(object):

    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """
        d = collections.defaultdict(list)
        for i in range(len(equations)):
            unkn1 = equations[i][0]
            unkn2 = equations[i][1]
            value = values[i]
            d[unkn1].append((value, unkn2))
            d[unkn2].append((1 / value, unkn1))

        print(d)

        res = []
        for u1, u2 in queries:
            print("u1: ", u1, "\tu2: ", u2)
            if u1 not in d or u1 not in d:
                res.append(-1)
                continue

            dictEquation1 = d[u1]
            queue = collections.deque()
            queue.append([u1, 1])
            visited = {u1: 1}
            ans = False
            while (queue):
                unkn, val = queue.popleft()
                if unkn == u2:  # point to itself, val = 1
                    ans = val
                    break
                else:
                    if unkn in d:
                        for v, u in d[unkn]:
                            if u not in visited:
                                queue.append([u, v * val])
                                visited[u] = 1
            if ans:
                res.append(ans)
            else:
                res.append(-1)

        return res


