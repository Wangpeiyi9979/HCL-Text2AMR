import penman
import re
from penman.graph import Graph
def get_deep_amr(amr):
    t = penman.parse(amr)
    var, branches = t.node
    branches = branches[1:]
    if len(branches) == 1 and len(branches[0][1]) ==1:
        return 2
    if len(branches) == 0:
        return 1
    deep = 0
    for branche in branches:
        role, target = branche
        try:
            sub_amr = penman.format(target)
            deep = max(deep, get_deep_amr(sub_amr) + 1)
        except:
            deep = max(deep, 2)
    return deep



def get_given_deep_amr(amr, Deepth=3):
    indent=3
    amr = penman.decode(amr)
    lines = penman.encode(amr, indent=indent)
    lines = lines.split('\n')
    instances = amr.instances()
    v2name = {}
    for ins in instances:
        v2name[ins[0]] = ins[-1]
    need_lines = []
    deeps = []
    miss_nodes = []
    for line in lines:
        if ':' in line:
            blank = line.split(':')[0]
        else:
            blank = ''
        deep_line = len(blank) // indent
        if deep_line <= Deepth:
            need_lines.append(line)
            deeps.append(deep_line)
        else: 
            node = re.findall('\(<.*/', line)
            if len(node):
                node = node[0][1:-1].strip()
                miss_nodes.append(node)
    for idx in range(len(need_lines)):
        now_line = need_lines[idx]
        now_deep = deeps[idx]
   
        if idx + 1 < len(need_lines):
            next_line = need_lines[idx+1]
            next_deep = deeps[idx+1]
            if now_line.endswith(')') or next_deep > now_deep: 
                continue
            if next_deep == now_deep:
                if ' (' in now_line:  
                    need_lines[idx] = need_lines[idx] + ' )'
            elif next_deep < now_deep:
                if ' (' in now_line:
                    add_deep = now_deep - next_deep +1
                else:
                    add_deep = now_deep - next_deep
                need_lines[idx] += ')' * add_deep
        else:
            if not now_line.endswith(')'):
                need_lines[idx] += ')' * (now_deep + 1)


    final_lines = []
    for line in need_lines:
        try:
            pointer = re.findall('<pointer-\d+>', line)[0]
        except:
            pointer = "<pointer-never>"
        if pointer in miss_nodes:
            line = line.replace(pointer, '( {} / {} )'.format(pointer, v2name[pointer]))
        final_lines.append(line)

    res_amr = '\n'.join(final_lines)
    return res_amr


def get_all_sub_triples(triples_res, node, all_triples, have_add):
    for triple in all_triples:
        if triple in triples_res:
            continue
        src, rel, tgt = triple
        if src == node:
            triples_res.add(triple)
            if tgt not in have_add and rel != ':instance':
                get_all_sub_triples(triples_res, tgt, all_triples, have_add)
                have_add.add(tgt)

def get_sub_amrs(amr, DEEP=4):
    g = penman.decode(amr)
    all_vars = set()
    for triples in g.triples:
        src, rel, tgt = triples
        if rel == ':instance':
            all_vars.add(src)
    all_amr = []
    for var in all_vars:
        triples_res = set()
        have_add = set()
        get_all_sub_triples(triples_res, var, g.triples, have_add)
        sub_amr  = penman.encode(Graph(triples_res), top=var)
        if get_deep_amr(sub_amr) >= DEEP:
            all_amr.append(sub_amr)
    return all_amr




if __name__ == '__main__':
    amr = """
    (u / underscore-01
    :ARG0 (e / experience-01
             :ARG0 (p / person
                      :wiki "Wang_Shi_(entrepreneur)"
                      :name (n / name
                               :op1 "Wang"
                               :op2 "Shi"))
             :ARG1 (c / condemn-01
                      :ARG0 (p2 / public)
                      :ARG1 p))
    :ARG1 (r / relation-03
             :ARG0 (c3 / class
                       :mod (r2 / rich)
                       :part-of (s / society
                                   :mod (c5 / country
                                            :wiki "China"
                                            :name (n2 / name
                                                      :op1 "China"))))
             :ARG2 (c4 / class
                       :mod (p3 / poor)
                       :part-of s)
             :ARG1-of (t / tense-03))
    :degree (e2 / extent
                :mod (c2 / certain)))
     """
    print(get_given_deep_amr(amr, 4))
    # assert False
    lines = open('../data/amrs/train.txt').readlines()
    amr = ""
    from tqdm import tqdm
    for line in tqdm(lines):
        if "# ::" in line:
            continue
        if line != '\n':
            amr = amr + " " + line
        else:
            if amr == "":
                continue
            for dp in range(1, get_deep_amr(amr)):
                sub_amr = get_given_deep_amr(amr, dp)
                if get_deep_amr(sub_amr) != dp + 1:
                    import ipdb; ipdb.set_trace()
            amr = ""
    
   