import pandas as pd 
import numpy as np
from random import shuffle
from math import log
from graphviz import Digraph
styles = {
    'leaf': {'shape': 'record', 'style': 'filled', 'color': 'yellow'},
    'crit': {'shape': 'record'},
}


class split:
    #Initialisierung des Splits
    def __init__(self, attribute, values , split_type, bound = None):
        
        self.attribute = attribute      # Name des Attributes  
        self.split_values = values            # Menge der Split values 
        self.type = split_type          # Einer der Werte 'categorical' oder 'numerical'
        self.bound = bound              # Wird nur angegeben  wenn  split_type == numerical 
        pass
     
    def status(self):
        print('\n Attribut:',self.attribute,'\n split_values:', self.split_values)
        pass
    
    def copy(self):
        copy_split = split(self.attribute,self.split_values.copy(), self.type, self.bound )
        
        return copy_split
        pass
    
    pass


class node:
     #Initialisierung des Nodes
    def __init__(self, nNr = None, nLabel = None, nType = None, data = None, n_split = None):
        
        self.node_nr = nNr
        self.label = nLabel       
        self.type = nType           # eine der Ausprägungen 'criterion'/'leaf'     
        self.subset = data  
        self.split = n_split        # 'categorical' oder 'numerical' (nur angeben, wenn node_type = 'criterion')
        
        self.parent = None
        self.child_nodes = []
        self.edges = []
    
    def status(self):
        children = []
        for i in self.child_nodes:
            children.append(i.node_nr)
        
        print('\n Nr:',self.node_nr, '\n Label:', self.label,'\n Type:', self.type, '\n Children:', children, '\n Data: \n', self.subset )
        
    def copy(self):
        
        copy_node = node(self.node_nr, self.label, self.type, self.subset.copy(), self.split)
        copy_node.parent = self.parent
        
        for nd in self.child_nodes:
            copy_node.child_nodes.append(nd.copy())
            copy_node.child_nodes[-1].parent = copy_node
        
        for ed in self.edges:
            copy_node.edges.append(ed.copy()) 
        
        return copy_node
    pass


class edge:
    #Initialisierung 
    def __init__(self, root_nr = None, target_nr = None, label = ''):
        self.root_nr = root_nr
        self.target_nr = target_nr       
        self.label = label     
    
    def status(self):
        print('\n Root:',self.root_nr,'\n Target:', self.target_nr, '\n Label:', self.label)
        
    def copy(self):
        copy_edge = edge(self.root_nr, self.target_nr, self.label)
        
        return copy_edge
    
    pass



def majority_value(data, target):
    try:
        majority_value = data[target].value_counts().idxmax() 
    except:
        majority_value = None
    return majority_value        
    pass


def distribution(data, target):
    
    return data[target].value_counts().sort_index().tolist()


def misclassification_error(frequencies):
    
    return (sum(frequencies) - max(frequencies)) / sum(frequencies)

#Entropiefunktion, Eingabe: absolute Häufigkeiten der einzelnen Werte der Zielvariable als Liste
def entropy(frequencies):
    try:
        total = sum(frequencies) 
        return sum([-p / total * log(p / total, 2) for p in frequencies])
    except:
        return 0


def information_gain(data, target, current_split):
    
    pre_entropy = entropy(distribution(data, target))
    
    distributions_of_subsets = []
  
    if current_split.type == 'categorical':
        for split_value in current_split.split_values:

            current_df = data[data[current_split.attribute] == split_value]
            distributions_of_subsets.append(distribution(current_df, target))
   
    elif current_split.type == 'numerical': 
        
            subset1 = data[data[current_split.attribute] <= current_split.bound]
            distributions_of_subsets.append(distribution(subset1, target))
            
            subset2 = data[data[current_split.attribute] > current_split.bound]
            distributions_of_subsets.append(distribution(subset2, target))
    
                 
    total_elements = data[target].count()
    post_entropy = 0
    
    for current_distribution in distributions_of_subsets:
        elements = sum(current_distribution)
        post_entropy += (elements/total_elements)*entropy(current_distribution)
    
    info_gain = pre_entropy - post_entropy
    
    return info_gain



def information_gain_ME(data, target, current_split):
    
    pre_ME = misclassification_error(distribution(data, target))
    
    distributions_of_subsets = []
    post_total_elements = 0
  
    if current_split.type == 'categorical':
        for split_value in current_split.split_values:

            current_df = data[data[current_split.attribute] == split_value]
            current_distr = distribution(current_df, target)
            distributions_of_subsets.append(current_distr)
            post_total_elements += sum(current_distr)
   
    elif current_split.type == 'numerical': 
        
            subset1 = data[data[current_split.attribute] <= current_split.bound]
            distr1 = distribution(subset1, target)
            distributions_of_subsets.append(distr1)
            post_total_elements += sum(distr1)
            
            subset2 = data[data[current_split.attribute] > current_split.bound]
            distr2 = distribution(subset2, target)
            distributions_of_subsets.append(distr2)
            post_total_elements += sum(distr2)
    
    post_ME = 0
    
    for current_distribution in distributions_of_subsets:
        elements = sum(current_distribution)
        post_ME += (elements/post_total_elements)*misclassification_error(current_distribution)
    
    info_gain = pre_ME - post_ME
    
    return info_gain



def find_all_splits(data, target):
    
    # Wir wollen alle möglichen Splits in einer Liste sammeln
    # Für jedes Attribut wird ein Split erstellt und in list_of_splits abgespeichert
    list_of_splits = [] 
    
    attribute_list_categorical = data.drop(target, axis = 1).select_dtypes(exclude = 'number').columns
    attribute_list_numerical = data.drop(target, axis = 1).select_dtypes(include = 'number').columns
    
    
    # Wir gehen der Reihe nach alle kategorialen Attribute durch
    for current_attribute in attribute_list_categorical:            
        
        # Die verschiedenen Werte in der Spalte des aktuellen Attributs... 
        # ... werden mittels der unique()-Funktion ausgelesen
        value_set = np.sort(data[current_attribute].dropna().unique())                                                      
        
        # Der aktuelle Split wird erstellt und er besteht aus Split-Attribut und Wertemenge 
        current_split = split(current_attribute, value_set, 'categorical') 
        
        list_of_splits.append(current_split)
        
        
     # Wir gehen der Reihe nach alle numerischen Attribute durch
    for current_attribute in attribute_list_numerical:            
    
        value_set = np.sort(data[current_attribute].dropna().unique())                                                      
        
        for i in range(len(value_set) - 1):
            current_bound = value_set[i]
            split_values = ['≤' + str(round(current_bound,2)), '>' + str(round(current_bound,2)) ]
            current_split = split(current_attribute, split_values, 'numerical', current_bound) 
            list_of_splits.append(current_split)        
                    
    #shuffle(list_of_splits)  

    return list_of_splits

    pass



def identify_best_split(data, target, list_of_splits, criterion = information_gain):
    
    best_split = None
    best_information_gain = -1
    
    for current_split in list_of_splits:
            
            current_information_gain = criterion(data, target, current_split)		
            
            if current_information_gain >= best_information_gain:
                best_information_gain = current_information_gain
                best_split = current_split
            
    return best_split
    #Output: Most productive split according to information gain
    pass


def apply_split(data, current_split):
    
    list_of_subsets = []
    
    if current_split.type == 'categorical':
        for current_value in current_split.split_values:

            current_df = data[data[current_split.attribute] == current_value]     #NaN Werte fallen weg
            #current_df.append(data[data[current_split.attribute].isna()])        #Alternativ: Umgang mit NaN einfach in jeden Ast weiterleiten                  
            list_of_subsets.append(current_df)
            
    elif current_split.type == 'numerical':
        subset1 = data[data[current_split.attribute] <= current_split.bound] #NaN Werte fallen weg
        list_of_subsets.append(subset1)
            
        subset2 = data[data[current_split.attribute] > current_split.bound]
        list_of_subsets.append(subset2)
            
    
    return list_of_subsets
    #Output: List of Subsets(DataFrames)
    pass

def import_data(tree, node_nr, data):

    if tree.tree_nodes[node_nr].type == 'leaf':
        tree.tree_nodes[node_nr].subset = data

    else:
        tree.tree_nodes[node_nr].subset = data
        children = tree.tree_nodes[node_nr].child_nodes
        attribute = tree.tree_nodes[node_nr].split.attribute
        
        if tree.tree_nodes[node_nr].split.type == 'numerical':
            bound = tree.tree_nodes[node_nr].split.bound
            tree = import_data(tree, children[0].node_nr, data[data[attribute] <= bound])
            tree = import_data(tree, children[1].node_nr, data[data[attribute] > bound])

        else:
            split_values = tree.tree_nodes[node_nr].split.split_values
            for i in range(len(split_values)):
                if split_values[i] == 'True':
                    if True in data[attribute].unique(): 
                        split_values[i] = True
                elif split_values[i] == 'False':
                    if False in data[attribute].unique(): 
                        split_values[i] = False

            for i in range(len(children)):
                tree = import_data(tree, children[i].node_nr, data[data[attribute] == split_values[i]])

            
    return tree

def import_tree(filename):

    treename = filename +'.tree'

    file = open(treename,'r', encoding='utf-8')
    tree_info = file.read().splitlines()
    file.close()

    data = pd.read_pickle(filename+'.pkl')
    target = tree_info[1]

    tree = DecisionTree(data = data, target = target)

    node_index = tree_info.index('NODES')
    edges_index = tree_info.index('EDGES')

    for i in tree_info[node_index+1:edges_index]:
        node_info = i.split(';')
        
        if node_info[2] == 'criterion':
            
            if node_info[5] == 'numerical':
                current_split = split(attribute = node_info[3], values = node_info[4].split(',') , split_type = node_info[5], bound = float(node_info[6]))
            else:
                current_split = split(attribute = node_info[3], values = node_info[4].split(',') , split_type = node_info[5], bound = None)  

            tree.tree_nodes[int(node_info[0])] = node(nNr = int(node_info[0]), nLabel = node_info[1], nType = node_info[2], n_split = current_split)

        else: 

            tree.tree_nodes[int(node_info[0])] = node(nNr = int(node_info[0]), nLabel = node_info[1], nType = node_info[2])
    

    for i in tree_info[edges_index+1:]:
        edge_info = i.split(';')
    

        tree.new_edge(root = int(edge_info[0]), target = int(edge_info[1]), label = edge_info[2])

    tree = import_data(tree, 1, data)

    return tree

class DecisionTree:
    
    #Initialisierung des Decision Tree
    def __init__(self, crit = None, target = None, data = pd.DataFrame(), target_values = None):
        
        self.tree_edges = []
        self.tree_nodes = {}
        self.tree_graph = Digraph()
        self.target = target
        self.data = data


        if (not self.data.empty) and (self.target != None):
            self.target_values = self.data[self.target].value_counts().sort_index().index.tolist()
            self.return_leaf_node(self.data, self.target)
        else:
            self.target_values=target_values


        if (crit == None) or (crit == 'entropy'):
        	self.criterion = information_gain
        elif crit == 'misclassification_error':
        	self.criterion = information_gain_ME
        else: 
            self.criterion = information_gain



        #self.depth = 0
        
        pass
    
    def grow_tree(self, data = pd.DataFrame(), target = None, crit = 'misclassification_error', max_depth = float('inf'), act_depth = 0, min_gain = 0, min_leaf_cases=0):
        
        if act_depth == 0:
            if target == None:
                if type(self.target) == type(None):
                    return 'target is missing'
                else:
                    target = self.target
            else:
                self.target = target

            
            if data.empty:
                if self.data.empty:
                    return 'data is missing'
                else:
                    data = self.data
       
            self.data = data.dropna(subset=[self.target]) 
            self.target_values = self.data[self.target].value_counts().sort_index().index.tolist()

            self.tree_edges = []
            self.tree_nodes = {}
            
            
            if crit == 'entropy':
                self.criterion = information_gain
            elif crit == 'misclassification_error':
                self.criterion = information_gain_ME
            else:
            	self.criterion = information_gain_ME

        attributes = (data.columns).drop(target)
        #print('loading...')
        
        if (data[target].nunique() == 1) or (len(attributes) == 0) or (act_depth >= max_depth) or (len(data) < min_leaf_cases):
            # Falls nur ein Wert für die Zielvariable vorliegt, gib ein leaf mit diesem Wert aus
            # Falls Anzahl der Attribute 0 ist, gib ein leaf mit dem Mehrheitswert der Zeilvariable aus
            # Falls maximale Tiefe des Baums erreicht ist, gib leaf mit Mehrheitswert der Zeilvariable aus 
            self.return_leaf_node(data, target)
        
        # Falls vorherige Abfragen nicht zutrafen wird ein weiterer Split gesucht um ihn anzuwenden
        else:
            #Finde alle möglichen Splits
            list_of_splits = find_all_splits(data, target)
            
            if len(list_of_splits) > 0:
                #Identifiziere den besten Split unter allen Splits
                best_split = identify_best_split(data, target, list_of_splits, self.criterion)
                

                # Überprüfen: Ist der Split produktiv?
                if information_gain(data, target, best_split) > min_gain: # Ist best_split produktiv?

                    #Wende den besten Split auf die Inputdaten an und erstelle somit ein Liste von Teildatensätzen
                    list_of_subsets = apply_split(data, best_split)

                    
                    #Den erstellten Split als Knoten ausgeben, falls best_split produktiv ist        
                    current_node = self.return_split_node(best_split, data)


                    #Rekursive weitere Anwendung für jeden erstellten Teildatensatz
                    for i in range(len(list_of_subsets)):
                        
                        next_node_nr = len(self.tree_nodes) + 1                        
                        new_input_subset = list_of_subsets[i].drop(best_split.attribute, axis = 1)

                        self.grow_tree(new_input_subset, target, self.criterion, max_depth, act_depth+1)
                        
                        self.new_edge(root = current_node.node_nr, target = next_node_nr, label = best_split.split_values[i])
                    


                else:
                    #Falls best_split nicht produktiv, dann leaf ausgeben
                    self.return_leaf_node(data, target)
                    pass
                
            else:
                #Falls list_of_splits leer ist gib einen 'leaf-node' aus #(Passiert, falls nur noch Split Attribute verbleiben 
                self.return_leaf_node(data, target)
                pass    
        pass
    
    
    def return_leaf_node(self, data, target):
        
        try: 
            node_nr = list(self.tree_nodes)[-1] + 1       # Neuer Knoten bekommt die nächst freie Nummer in tree_nodes
        except:
            node_nr = 1

        node_label = majority_value(data, target) #Neuer Knoten bekommt als Label den Mehrheitswert für target

        current_node = node(node_nr, node_label, 'leaf', data) # Knoten wird erstellt
        
        self.tree_nodes[node_nr] = current_node       # Knoten wird zum dictionary aller Knoten hinzugefügt
    
    
    
    def return_split_node(self, best_split, data):
        try: 
            node_nr = list(self.tree_nodes)[-1] + 1       # Neuer Knoten bekommt die nächst freie Nummer in tree_nodes
        except:
            node_nr = 1

        node_label = best_split.attribute       # Das Label des Knotens ist das aktuelle Split Attribut
        
        current_node = node(node_nr, node_label, 'criterion', data, best_split) # Knoten wird erstellt
        
        self.tree_nodes[node_nr] = current_node       # Knoten wird zum dictionary aller Knoten hinzugefügt
        
        return self.tree_nodes[node_nr]

    
    def new_edge(self, root, target, label):
        
        #Edge bauen
        new_edge = edge(root, target, str(label))                                            # N E U
        self.tree_edges.append(new_edge)
        
        #Nodes informieren
        nd_root = self.tree_nodes[root]
        nd_target = self.tree_nodes[target]
            
        nd_root.child_nodes.append(nd_target)
        nd_root.edges.append(new_edge)
        
        nd_target.parent = nd_root
        nd_target.edges.append(new_edge)
    
    def get_data(self, node_nr):
        return self.tree_nodes[node_nr].subset
        pass

    def get_target_distribution(self, node_nr):
        distribution = self.tree_nodes[node_nr].subset[self.target].value_counts()

        if len(distribution) < len(self.target_values):
            for i in self.target_values:
                try:
                    distribution[i]
                except:
                    if not type(i) == bool:
                        distribution[i]=0
                    else:
                        index = list(distribution.index)
                        index.append(i)
                        values = list(distribution.values)
                        values.append(0)
                        distribution = pd.Series(index=index, data= values)

        return distribution.sort_index()
        pass


    def get_node_count(self):
    	return len(self.tree_nodes)

    def get_leaf_node_count(self):
        count=0

        for i in self.tree_nodes:
            if self.tree_nodes[i].type == 'leaf':
                count+=1

        return count

    def get_mean_leaf_cases(self):
        count = 0
        cases = 0

        for i in self.tree_nodes:
            if self.tree_nodes[i].type == 'leaf':
                count+=1
                cases+=len(self.tree_nodes[i].subset)

        return round(cases/count,2)


    def get_max_depth(self, current_depth = 0, max_depth = 0, current_node = None):

    	if current_node == None:
    		current_node = self.tree_nodes[1]

    	current_depth+=1
    	if max_depth < current_depth:
    		max_depth = current_depth

    	if current_node.type == 'leaf':
    		return max_depth

    	else:
    		for node in current_node.child_nodes:
    			max_depth = self.get_max_depth(current_depth = current_depth, max_depth = max_depth, current_node = node)

    	return max_depth
    	pass
    
    def get_status(self):
        leaf_node_count = self.get_leaf_node_count()

        status = {'node_count' : int(self.get_node_count()),
        'leaf_node_count' : int(leaf_node_count),
        'max_depth' : int(self.get_max_depth()),
        'mean_leaf_cases' : round(len(self.tree_nodes[1].subset) / leaf_node_count,2)}

        return pd.Series(status)


    def query(self, input_series, confidence = False):
        #Input: Series, die einen Wert für jedes Attribut enthält
        
        current_node = self.tree_nodes[1]
        next_nr = current_node.node_nr
        
        # Wir gehen so lange durch den Baum, bis wir in einem 'leaf-node' sind
        while current_node.type == 'criterion':
            
            #Prüfwert um später zu schauen, ob eine neuer Knoten gefunden wurde (wird nicht gefunden falls NaN)
            old_nr = next_nr
            
            #prüfen ob der Split 'categorical' oder 'numerical' ist
            if current_node.split.type == 'categorical':
    
                # Suche die Kante, die am aktuellen Knoten liegt und zum Wert der Input Series passt
                for edge in current_node.edges: 
                    if (edge.label == str(input_series[current_node.label])):
                        next_nr = edge.target_nr
                        break
                
                #Prüfen ob neuer Knoten gefunden wurde
                #Falls der im Kriterum abgefragte Wert fehlt (NaN) kann kein Ast ausgewählt werden
                #der Baum gibt den Mittelwert/Mehrheitswert im aktuellen Kriteriums-Knoten aus 
                if old_nr == next_nr:

                    if confidence:
                        value = majority_value(current_node.subset, self.target)
                        value_conf = str(round(distribution(current_node.subset, self.target)*100,1))+'%'
                        return value, value_conf
                    else:
                        return majority_value(current_node.subset, self.target)
                    
                
                # der nächste Knoten wir gesucht und als current_node gespeichert, um in die nächste Iteration der Schleife zu gehen 
                for nd in current_node.child_nodes:
                    if nd.node_nr == next_nr:
                        current_node = nd
                        break
            
            elif current_node.split.type == 'numerical':

                if (input_series[current_node.label] <= current_node.split.bound):
                    current_node = current_node.child_nodes[0]
                else:
                    current_node = current_node.child_nodes[1]

        if confidence:
            value = current_node.label
            current_distr = distribution(current_node.subset, self.target)
            value_conf = str(round((max(current_distr)/sum(current_distr))*100,1))+'%'
            return value, value_conf
        else:  
            return current_node.label
        
        pass

    
    
    def prediction_accuracy(self, data, detailed = False , conf_mat = False, conf_mat_row_perc = False, conf_mat_col_perc = False ):
        #print('loading...')
        data = data.dropna(subset=[self.target]).reset_index(drop=True)
        targets = data[self.target]

        prediction_list = []
        
        #berechne alle outputs/predictions
        for i in data.index:
            prediction_list.append(self.query(data.loc[i]))

        predictions = pd.Series(prediction_list, name = 'prediction')
        predictions.index = targets.index
        

        accuracy = (predictions == targets).mean()
        
     
        
        if conf_mat or conf_mat_row_perc or conf_mat_col_perc or detailed:
            #Berechne Fehler verschiedener Art und gebe in crosstable aus
            df_evaluation = pd.concat([targets, predictions], axis = 1)

            values = np.sort(targets.append(predictions).unique())
            columns= pd.Series(values, name = 'prediction')
            index = pd.Series(values, name = 'correct')
            
            df_crosstable_rel1 = pd.DataFrame(columns=columns, index= index)
            df_crosstable_rel2 = pd.DataFrame(columns=columns, index= index)
            df_crosstable_abs = pd.DataFrame(columns=columns, index= index) 

            for i in values:
                current_eval1 = df_evaluation[df_evaluation[self.target] == i]
                current_eval2 = df_evaluation[df_evaluation['prediction'] == i]
                for j in values:
                    current_rate1 = str(round((current_eval1['prediction'] == j).mean()*100,1))+'%'
                    current_rate2 = str(round((current_eval2[self.target] == j).mean()*100,1))+'%'
                    current_sum = (current_eval1['prediction'] == j).sum()
                    df_crosstable_rel1[j][i] = current_rate1
                    df_crosstable_rel2[i][j] = current_rate2
                    df_crosstable_abs[j][i] = current_sum
            if conf_mat:
             	display(df_crosstable_abs)
            if conf_mat_row_perc: 
             	display(df_crosstable_rel1)
            if conf_mat_col_perc:
             	display(df_crosstable_rel2)
            if detailed:
            	display(df_crosstable_abs)
            	display(df_crosstable_rel1)
            	display(df_crosstable_rel2)

        return accuracy
        
        pass

    def prediction_data(self, data, node_nr = 1, list_subsets = []):
        #gibt eine Liste mit allen finalen Teildatensätzen und deren Predictions zurück
        current_list = list_subsets
        current_data = data

        # Ausgaben für alle Zeilen mit fehlenden Werten (nan)
        if node_nr == 1:
            current_list = []
            nan_data = current_data[current_data.isna().any(axis=1)]#.reset_index(drop=True)

            prediction_list = []
            #berechne alle outputs/predictions
            for i in nan_data.index:
                prediction_list.append(self.query(data.loc[i]))
            
            nan_data['Prediction'] = prediction_list
            
            current_list.append(nan_data)
            current_data = current_data.dropna()

        

        if self.tree_nodes[node_nr].type == 'leaf':
            current_data['Prediction'] = self.tree_nodes[node_nr].label
            current_list.append(data)
            return current_list

        else:

            children = self.tree_nodes[node_nr].child_nodes
            next_split = self.tree_nodes[node_nr].split
            if next_split.type == 'numerical':
                next_data1 = current_data[current_data[next_split.attribute]<=next_split.bound] 
                next_data2 = current_data[current_data[next_split.attribute]>next_split.bound] 

                current_list = self.prediction_data(data = next_data1, node_nr = children[0].node_nr, list_subsets = current_list)
                current_list = self.prediction_data(data = next_data2, node_nr = children[1].node_nr, list_subsets = current_list)

            else:
                for i in range(len(children)):
                    next_data = current_data[current_data[next_split.attribute] == next_split.split_values[i]] 
                    next_nr = children[i].node_nr
                    current_list = self.prediction_data(data = next_data, node_nr = next_nr, list_subsets = current_list)
            
            
        
            return current_list
           
        pass

    def prediction_accuracy2(self, data, detailed = False):
        
        data = data.dropna(subset=[self.target]).reset_index(drop=True)
        subset_list = self.prediction_data(data)

        data_predicted = subset_list[0]
        for i in subset_list[1:]:
            data_predicted = data_predicted.append(i)
           
        accuracy = (data_predicted['Prediction'] == data_predicted[self.target]).mean()

        return accuracy

    def evaluate_fairness(self, data, fairness_attribute, detailed = False):
    	subsets = apply_split(data, split(attribute = fairness_attribute, values = data[fairness_attribute].unique(), split_type='categorical') )

    	for i in subsets:
    		print(i[fairness_attribute].unique()[0])
    		if detailed:
    			self.prediction_accuracy(i, detailed = detailed)
    		else:
    			print(str(round(self.prediction_accuracy(i, detailed = detailed)*100,1))+'%')  

    
    def evaluation_node_count(self, train_data, test_data):
    	 
    	node_count= self.get_node_count()
    	max_depth= self.get_max_depth()
    	
    	list_of_trees=[]

    	for i in range(1,max_depth+1):
    		help_tree = self.copy()
    		help_tree.prune_depth(i)
    		list_of_trees.append(help_tree)

    	index = []

    	for i in list_of_trees:
    		index.append(i.get_node_count())

    	df = pd.DataFrame(index=index) 	
    	
    	train_list=[]
    	for j in list_of_trees:
    		train_list.append(j.prediction_accuracy(train_data))
    	df['Train Data'] = train_list

    	test_list=[]
    	for j in list_of_trees:
    		test_list.append(j.prediction_accuracy(test_data))
    	df['Test Data'] = test_list

    	return df

    def evaluation_depth(self, train_data, test_data):
	
    	node_count= self.get_node_count()
    	max_depth= self.get_max_depth()
    	
    	list_of_trees=[]

    	for i in range(1,max_depth+1):
    		help_tree = self.copy()
    		help_tree.prune_depth(i)
    		list_of_trees.append(help_tree)

    	index = range(1,max_depth+1)

    	df = pd.DataFrame(index=index)
    	
    	train_list=[]
    	for j in list_of_trees:
    		train_list.append(j.prediction_accuracy(train_data))
    	df['Train Data'] = train_list

    	test_list=[]
    	for j in list_of_trees:
    		test_list.append(j.prediction_accuracy(test_data))
    	df['Test Data'] = test_list

    	return df

    def prune_node(self, prune_node_nr, prune_node = None):
        
        if prune_node == None:
            prune_node = self.tree_nodes[prune_node_nr]
            
                    
        if prune_node.parent == None:
            root_node = node()
        else:
            root_node = prune_node.parent
        
        list_of_children = prune_node.child_nodes
        prune_node.child_nodes = []
                
        if (prune_node.type == 'leaf') & (root_node.type == 'leaf'):
            
            del self.tree_nodes[prune_node_nr]
            
            for edge in prune_node.edges:
                if edge.target_nr == prune_node_nr:
                    self.tree_edges.remove(edge)
                    #root_node.edges.remove(edge)                                        # Warum geht das nicht so?
                    for ed in root_node.edges:
                        if ed.target_nr == prune_node_nr:
                            root_node.edges.remove(ed)
                    
        
        else:
            prune_node.type = 'leaf'
            prune_node.label = majority_value(prune_node.subset, self.target)
            
            
            for child in list_of_children:
                self.prune_node(child.node_nr, child)
                
            
            if root_node.type == 'leaf':
                del self.tree_nodes[prune_node_nr]
            
                for edge in prune_node.edges:
                    if edge.target_nr == prune_node_nr:
                        self.tree_edges.remove(edge)
                        #root_node.edges.remove(edge)
                        for ed in root_node.edges:
                            if ed.target_nr == prune_node_nr:
                                root_node.edges.remove(ed)
               
        pass

    def prune_depth(self, max_depth, current_depth = 0, current_node = None):

    	if current_node == None:
    		current_node = self.tree_nodes[1]

    	current_depth+=1

    	if current_depth == max_depth:
    		self.prune_node(current_node.node_nr)

    	else:
    		for node in current_node.child_nodes:
    			self.prune_depth(max_depth = max_depth, current_depth = current_depth, current_node = node)

    	pass

    def validation_pruning(self, validation_sample, root_node = None):                     #root_nr = 1
        
        if root_node == None:
            current_node = self.tree_nodes[1]
        else:
            current_node = root_node

        if current_node.type == 'leaf':
            
            pass

        else:    

            for child in current_node.child_nodes:                                                                    
                test_tree = self.validation_pruning(validation_sample, child)

            print('Test Node:', current_node.node_nr)
            
            if test_tree == None:
                test_tree = self.copy()
            
            pre_test_accuracy = test_tree.prediction_accuracy2(validation_sample)
            
            test_tree.prune_node(current_node.node_nr)

            post_test_accuracy = test_tree.prediction_accuracy2(validation_sample)
            
            if post_test_accuracy >= pre_test_accuracy:
                self.prune_node(current_node.node_nr, current_node)
                print('Prune Node:',current_node.node_nr)
                print('Node-Count', len(self.tree_nodes))
                print('New Test-Score', post_test_accuracy)
                return test_tree
                
            return None
        pass
    def manual_split(self, attribute = 'auto', threshold = 'auto', node_nr = 1, doc = False):
        
        documentation = False
        list_of_splits = []

        if node_nr == 1:
            try:
                self.tree_nodes[1]
            except:
                self.tree_nodes[1] = node( nNr = 1, data = self.data)
        
        if self.tree_nodes[node_nr].type == 'criterion':
            self.prune_node(node_nr)
        
        self.tree_nodes[node_nr].type = 'criterion'
        
        if attribute == 'auto':

            list_of_splits = find_all_splits(self.tree_nodes[node_nr].subset, self.target)
            best_split = identify_best_split(self.tree_nodes[node_nr].subset, self.target, list_of_splits, self.criterion)
            self.tree_nodes[node_nr].label = best_split.attribute

        else:

            self.tree_nodes[node_nr].label = attribute
            
            if self.tree_nodes[node_nr].subset[attribute].dtype in ['float64','int64', int, float,'float32','int32']:
            
                if (type(threshold) in ['float64','int64', int, float,'float32','int32']):

                    if (threshold >= self.tree_nodes[node_nr].subset[attribute].min()) and (threshold < self.tree_nodes[node_nr].subset[attribute].max()):
                        split_values = ['≤' + str(round(threshold,2)), '>' + str(round(threshold,2)) ]
                        best_split = split(attribute, split_values, 'numerical', threshold)

                    else:
                        #list_of_splits = find_all_splits(self.tree_nodes[node_nr].subset[[self.target,attribute]], self.target)
                        #best_split = identify_best_split(self.tree_nodes[node_nr].subset, self.target, list_of_splits, self.criterion)
                        

                        if (threshold < self.tree_nodes[node_nr].subset[attribute].min()):
                            print('Error: Threshold value could not be used - too small \n')
                            self.tree_nodes[node_nr].type = 'leaf'
                            self.tree_nodes[node_nr].label = majority_value(self.tree_nodes[node_nr].subset, self.target)
                            return 'Error'
                            #print(str(best_split.bound)+' was used instead')
                        else:
                            print('Error: Threshold value could not be used - too large \n')
                            self.tree_nodes[node_nr].type = 'leaf'
                            self.tree_nodes[node_nr].label = majority_value(self.tree_nodes[node_nr].subset, self.target)
                            return 'Error'
                            #print(str(best_split.bound)+' was used instead')
                else:
                    list_of_splits = find_all_splits(self.tree_nodes[node_nr].subset[[self.target,attribute]], self.target)
                    best_split = identify_best_split(self.tree_nodes[node_nr].subset, self.target, list_of_splits, self.criterion)
                   
            else:
                value_set = self.tree_nodes[node_nr].subset[attribute].dropna().unique()                                                       
                best_split = split(attribute, value_set, 'categorical')
            
        list_of_subsets = apply_split(self.tree_nodes[node_nr].subset, best_split)
        for i in range(len(list_of_subsets)):
            next_node_nr = list(self.tree_nodes)[-1] + 1
            self.return_leaf_node(list_of_subsets[i].drop(best_split.attribute, axis = 1), self.target)
            self.new_edge(root = node_nr, target = next_node_nr, label = best_split.split_values[i])

        self.tree_nodes[node_nr].split = best_split      

        #Dokumentation aller splits, falls Parameter doc == True
        if doc == True and len(list_of_splits)>0:
            df_doc = pd.DataFrame(columns=['Attribute', 'Threshold', 'Criterion'])
            for i in range(len(list_of_splits)):
                current_split = list_of_splits[i]
                df_doc.loc[i] = pd.Series(index = ['Attribute', 'Threshold', 'Criterion'], data = [current_split.attribute, current_split.bound, round(self.criterion(self.tree_nodes[node_nr].subset, self.target, current_split),3)])
            return df_doc.sort_values('Criterion', ascending = False).reset_index(drop = True)
        pass

    def reset_node_index(self):
        
        new_tree_nodes={}
        help_dict={}
        index=1
        for j in self.tree_nodes:
            new_tree_nodes[index]=self.tree_nodes[j]
            new_tree_nodes[index].node_nr = index
            help_dict[j]=index
            index+=1

        for edge in self.tree_edges:
            edge.root_nr = help_dict[edge.root_nr]
            edge.target_nr = help_dict[edge.target_nr]

        self.tree_nodes = new_tree_nodes
        self.print_tree()

        return self.tree_graph

    
    def print_tree(self):
        
        self.tree_graph = Digraph(filename = ('Tree_' + str(self.target)))
        #self.tree_graph.attr(label = ('Zielvariable:' + self.target + '\n Verteilung:' + str(self.tree_nodes[1].subset[self.target].value_counts().sort_index().index.tolist())), labelloc = "t")
        self.tree_graph.attr(label = '''<<TABLE BORDER="1" CELLSPACING="2" CELLPADDING="2" BGCOLOR="lightblue">
                                            <TR>
                                                <TD><FONT POINT-SIZE="12.0" COLOR="black">''' + str(self.target) + '?' + '   ' + '''</FONT></TD>
                                            </TR>
                                            <TR>
                                                <TD><FONT  POINT-SIZE="10.0" COLOR="dimgrey">'''+ str(self.target_values) +'''</FONT></TD>
                                            </TR>
                                        </TABLE>>''' , labelloc = 't', labeljust = 'l')
        for current_node in list(self.tree_nodes.values()):
            if current_node.type == 'criterion':
                self.tree_graph.node(str(current_node.node_nr), '''<<TABLE BORDER="0" CELLSPACING="0" CELLPADDING="0.5">
                                                                <TR>
                                                                    <TD><FONT POINT-SIZE="10.0" COLOR="dimgrey">'''+ str(self.get_target_distribution(current_node.node_nr).tolist()) +'''</FONT></TD>
                                                                </TR>

                                                                <TR>
                                                                    <TD><FONT  POINT-SIZE="14.0" COLOR="black">'''+ str(current_node.label) + '?' + '''</FONT></TD>
                                                                </TR>
                                                                <TR>
                                                                    <TD ALIGN="RIGHT"><FONT POINT-SIZE="10.0" COLOR="dimgrey">'''+ 'Nr.' + str(current_node.node_nr) +'''</FONT></TD>
                                                                </TR>
                                                                    </TABLE>>''', styles['crit'])


            elif current_node.type == 'leaf':
                self.tree_graph.node(str(current_node.node_nr), '''<<TABLE BORDER="0" CELLSPACING="0" CELLPADDING="1">
                                                                <TR>
                                                                    <TD><FONT POINT-SIZE="10.0" COLOR="dimgrey">'''+ str(self.get_target_distribution(current_node.node_nr).tolist()) +'''</FONT></TD>
                                                                </TR>

                                                                <TR>
                                                                    <TD><FONT  POINT-SIZE="14.0" COLOR="black">'''+ str(current_node.label) + '''</FONT></TD>
                                                                </TR>
                                                                <TR>
                                                                    <TD ALIGN="RIGHT"><FONT POINT-SIZE="10.0" COLOR="dimgrey">'''+ 'Nr.' + str(current_node.node_nr) +'''</FONT></TD>
                                                                </TR>
                                                                    </TABLE>>''', styles['leaf'])    

        for current_edge in self.tree_edges:
                
                if self.tree_nodes[current_edge.root_nr].split.bound == None:
                	self.tree_graph.edge(str(current_edge.root_nr), str(current_edge.target_nr), current_edge.label, labeltooltip = current_edge.label)
                else:
                	self.tree_graph.edge(str(current_edge.root_nr), str(current_edge.target_nr), current_edge.label, labeltooltip = str(self.tree_nodes[current_edge.root_nr].split.bound))

        #self.tree_graph.view()
        return self.tree_graph
        
        pass
        
    def copy(self, copy_tree = None, current_node = None):
        if copy_tree == None:
            copy_tree = DecisionTree(target = self.target, data = self.data.copy(), target_values = self.target_values.copy())
            current_node = self.tree_nodes[1].copy()
        
        copy_tree.tree_nodes[current_node.node_nr] = current_node
        for edge in current_node.edges:
            if edge.target_nr == current_node.node_nr:
                copy_tree.tree_edges.append(edge)
                
        if len(current_node.child_nodes) == 0:
            pass
        else:
            for nd in current_node.child_nodes:
                copy_tree = self.copy(copy_tree, nd)
        
        
        return copy_tree
        pass
        
    def export(self, filename):

        treename = filename +'.tree'

        file = open(treename,'wb')
        
        #schreibe allgemeine Informationen über den Tree
        file.write('TREE\n'.encode("utf8"))
        encoded_unicode = self.target.encode("utf8")
        file.write(encoded_unicode)
        file.write('\n'.encode("utf8"))

        #schreibe Informationen über die Nodes
        file.write('NODES\n'.encode("utf8"))

        for n in self.tree_nodes.values():
            split_values = '' 
            
            if not n.split==None:
                for i in n.split.split_values:
                    split_values += str(i)
                    split_values += ','

                split_values = split_values[:-1]
                split_info = n.split.attribute + ';' + split_values + ';' + n.split.type + ';' + str(n.split.bound)
            else:
                split_info = str(None)

            node_info = str(n.node_nr) + ';' + str(n.label) + ';' + n.type + ';' + split_info
            encoded_unicode = node_info.encode("utf8")
            file.write(encoded_unicode)
            file.write('\n'.encode("utf8"))

        #schreibe Informationen über die Edges
        file.write('EDGES\n'.encode("utf8"))
        for n in self.tree_edges:    
            edge_info= str(n.root_nr) + ';' + str(n.target_nr) + ';' + str(n.label)
            encoded_unicode = edge_info.encode("utf8")
            file.write(encoded_unicode)
            file.write('\n'.encode("utf8"))

        file.close()

        self.data.to_pickle(filename+'.pkl')

        pass