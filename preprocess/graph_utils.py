from collections import defaultdict
import copy
import json

def split_and_handle_duplicate_nodes_in_linearized_penman(linearized_penman):
    splitted_linearized_penman = linearized_penman.split()
    map_node_id = defaultdict(lambda: 0)
    for i in range(len(splitted_linearized_penman)):
        item = splitted_linearized_penman[i]
        if (item != '(' and item != ')' and ':' not in item): # node
            node_with_id = item + '-' + str(map_node_id[item])
            map_node_id[item]+=1
            splitted_linearized_penman[i] = node_with_id      
    return splitted_linearized_penman

def convert_linearized_penman_to_tree(linearized_penman):
    splitted_linearized_penman = split_and_handle_duplicate_nodes_in_linearized_penman(linearized_penman)

    adj_list = defaultdict(list)
    stack = []
    set_visited_node = set()
    root = None

    curr_edge = None

    for item in splitted_linearized_penman:
        if (item[0]==':'): # edge
            curr_edge = item
        elif (item==')'):
            if (len(stack)>0):
                stack.pop()
        elif (item=='('):
            continue # do nothing
        else: # node
            if (len(stack)!=0):
                top_stack = stack[-1]
                if (item in set_visited_node):
                    continue

                adj_list[top_stack].append((curr_edge, item))
                stack.append(item)
                set_visited_node.add(item)
            else:
                root = item
                stack.append(item)  
                set_visited_node.add(item)
    return root, adj_list

def dfs_tree(root, adj_list, mode="dfs"):  # mode = linearized_penman/dfs/nodes_only
    path = []
    list_level = []

    def _dfs_tree_recurr(node_now, parent,  level=1):
        if (mode=='linearized_penman'):
            path.append('(')
            list_level.append(0) 
        path.append(node_now)
        list_level.append(level)
        for (edge, node_next) in adj_list[node_now]:
            if (node_next!=parent):
                if (mode!="nodes_only"):
                    path.append(edge)
                    list_level.append(level+1)
                _dfs_tree_recurr(node_next, node_now, level+1)
        if (mode=='linearized_penman'):
            path.append(')')
            list_level.append(0) 

    _dfs_tree_recurr(root, root)
    return path, list_level

def grammar_based_tree_traversal(root, adj_list):
    with open('preprocess/traversal_rule.json') as f:
        list_rule = json.load(f)['list_rule']
    
    path = []

    def _put_edge_node_in_path(pair_edge_node):
        if pair_edge_node not in path:
            path.append(pair_edge_node)
    
    def _traversal_tree_recurr(node_now, prev_edge=':root'):
        if (node_now not in adj_list):  # leaf node
            _put_edge_node_in_path((prev_edge, node_now))

        list_pair_edge_node = adj_list[node_now]
        priority_list_pair_edge_node = []
        is_picked = [False for _ in range(len(list_pair_edge_node))]
        for rule in list_rule:
            edge_condition = rule['condition']['edge']
            priority = rule['priority']
            for i, (edge, node) in enumerate(list_pair_edge_node):
                if edge_condition == edge:
                    is_picked[i] = True
                    if priority == "dependent_node":
                        _traversal_tree_recurr(node, edge)
                        _put_edge_node_in_path((prev_edge, node_now))
                    else: # head node
                        _put_edge_node_in_path((prev_edge, node_now))
                        _traversal_tree_recurr(node, edge)

        # traverse other node that is not in priority
        for i in range(len(is_picked)):
            if (not is_picked[i]):
                (edge, node) = list_pair_edge_node[i]
                _put_edge_node_in_path((prev_edge, node_now))
                _traversal_tree_recurr(node, edge)
    _traversal_tree_recurr(root)
    return path

def convert_linearized_penman_to_rule_based_traversal(linearized_penman):
    root, adj_list = convert_linearized_penman_to_tree(linearized_penman)
    path = grammar_based_tree_traversal(root, adj_list)

    str_path = ""
    for (edge, node) in path:
        node_without_id = node.split('-')[0]
        if (edge==None):
            edge = ':mod'
        str_path+= edge + " " + node_without_id + " "
    return str_path.strip()

def convert_linearized_penman_to_traversal_with_tree_level(linearized_penman, mode='dfs'):
    root, adj_list = convert_linearized_penman_to_tree(linearized_penman)
    path, list_level = dfs_tree(root, adj_list, mode)

    str_path = ""
    str_level = ""
    for i in range(len(path)):
        item = path[i]
        if (item==None):    #edge none
            item = ':mod'
        elif (item[0]!=':'):
            item = item.split('-')[0]   #node without id
        str_path+= item + " "
        str_level+= str(list_level[i]) + " "
    
    return str_path.strip(), str_level.strip()
    

if __name__=="__main__":
    # print(convert_linearized_penman_to_tree('( pergi :ARG0 ( kami :mod ( keluarga ) ) :ARG1 ( tamasya ) :time ( hari :mod ( minggu ) ) )'))    
    # print(convert_linearized_penman_to_tree('( pergi :ARG0 ( kami :mod ( keluarga ) ) :ARG1 ( tamasya ) :time ( hari :mod ( tamasya ) ) )'))    

    # root, adj_list = convert_linearized_penman_to_tree('( pergi :ARG0 ( kami :mod ( keluarga ) ) :ARG1 ( tamasya ) :time ( hari :mod ( minggu ) ) )')
    # # print(dfs_tree(root, adj_list))

    # print(grammar_based_tree_traversal(root, adj_list))
    print(convert_linearized_penman_to_traversal_with_tree_level('( pergi :ARG0 ( kami :mod ( keluarga ) ) :ARG1 ( tamasya ) :time ( hari :mod ( tamasya ) ) )', mode='linearized_penman'))

    # print(convert_linearized_penman_to_dfs_with_tree_level('( adik  ajak :ARG0 ( ibu ) :ARG1 a :location ( pasar ) )'))