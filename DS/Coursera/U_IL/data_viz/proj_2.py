import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def parse_csv_with_strs(path):
    temp = pd.read_csv(path, sep='|', header=None)
    temp = temp[0].str.split(' ', expand=True)
    del temp[0]
    return temp.apply(lambda row: ' '.join(filter(None, row.values)), axis=1)


def main():
    wiki_df = pd.read_csv(r'DS\Coursera\U_IL\data_viz\wiki-topcats.txt.gz', sep=' ', header=None)
    wiki_df.columns = ['from', 'to']
    names_df = parse_csv_with_strs(r'DS\Coursera\U_IL\data_viz\wiki-topcats-page-names.txt.gz')

    graph = nx.from_pandas_edgelist(wiki_df, source='from', target='to')
    nx.draw_networkx(graph)
    plt.show()




    


if __name__ == '__main__':
    main()