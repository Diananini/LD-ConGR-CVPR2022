def Jsi(gts, ps, label):
    g_set = set()
    p_set = set()
    for seg in gts:
        seg_points, g_l = seg
        s, e = seg_points
        if g_l == label:
            g_set.update(range(s, e + 1, 1))
    for seg in ps:
        seg_points, p_l = seg
        s, e = seg_points
        if p_l == label:
            p_set.update(range(s, e + 1, 1))
    inter_set = g_set & p_set
    union_set = g_set | p_set
    inter_v = len(inter_set)
    union_v = len(union_set)
    if union_v == 0:
        jsi = 0
    else:
        jsi = float(inter_v) / float(union_v)
        # if jsi > 0.6:
        #     return 1.
        # elif jsi < 0.2:
        #     return 0.
        # else:
        #     return jsi
    return jsi
