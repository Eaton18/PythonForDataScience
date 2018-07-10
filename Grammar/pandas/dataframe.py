
import pandas as pd

print(list('abcd'))

dfmi = pd.DataFrame([list('abcd'),
                     list('efgh'),
                     list('ijkl'),
                     list('mnop')],
                    columns=pd.MultiIndex.from_product([['one','two'],
                                                        ['first','second']]))

print(dfmi)

