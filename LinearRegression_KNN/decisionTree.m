function [new_accu_gini, train_accu_gini, new_accu_entro, train_accu_entro] = decisionTree(train_data, train_label, new_data, new_label, K)

    tree_gini = ClassificationTree.fit(train_data, train_label, 'MinLeaf', K, 'SplitCriterion', 'gdi', 'Prune', 'off');
    tree_entro = ClassificationTree.fit(train_data, train_label, 'MinLeaf', K, 'SplitCriterion', 'deviance', 'Prune', 'off');
    
    label_gini_train = tree_gini.predict(train_data);
    label_entro_train = tree_entro.predict(train_data);
    label_gini_new = tree_gini.predict(new_data);
    label_entro_new = tree_entro.predict(new_data);
    
    gini_train_sum = sum(label_gini_train == train_label);
    entro_train_sum = sum(label_entro_train == train_label);
    gini_new_sum = sum(label_gini_new == new_label);
    entro_new_sum = sum(label_entro_new == new_label);
    
    new_accu_gini = gini_new_sum / length(new_label);
    new_accu_entro = entro_new_sum / length(new_label);
    train_accu_gini = gini_train_sum / length(train_label);
    train_accu_entro = entro_train_sum / length(train_label);

end