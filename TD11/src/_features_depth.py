def features_depth(model, depth, acc=False):
    """Génère les indices des caractéristiques utilisées dans un arbre.

    L'argument `model` est l'arbre. Les indices sont générés
    uniquement à la profondeur `depth` sauf si `acc` est vrai. Dans ce
    cas, toutes les caractéristiques jusqu'à la profondeur `depth`
    sont générées.

    """

    tree = model.tree_
    def gen_id(i, depth):
        if tree.feature[i] >= 0:
            if acc or depth == 0:
                yield tree.feature[i]
        if depth != 0:
            yield from gen_id(tree.children_left[i], depth - 1)
            yield from gen_id(tree.children_right[i], depth - 1)

    yield from gen_id(0, depth)
