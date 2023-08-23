import graphviz
#import ipdb; ipdb.set_trace()
dot = graphviz.Digraph()  
norm_color = lambda x: str(int(x*5+6))
dot.attr('edge', colorscheme='rdylgn11')
dot.attr('node', colorscheme='x11')
dot.node('A', '1')
dot.node('B', '2', fillcolor='darkgoldenrod2', style='filled')
dot.node('C', '1', fillcolor='darkolivegreen2', style='filled')
dot.edge('B', 'A', weight='10')
dot.edge('C', 'A', weight='0.5', color=norm_color(0), label ='0.0')
dot.edge('C', 'B', weight='-0.3')

with open('tree.dot', 'w') as f:
    f.write(dot.source)
