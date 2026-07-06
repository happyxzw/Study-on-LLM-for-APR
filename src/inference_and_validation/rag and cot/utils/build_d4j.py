def build_d4j1_2():
    import json, re

    # 读取 defects4j_c1.json 中已用作测试集的项目，从知识库中排除
    c1_path = "/home/chenshiping/peft4apr/datasets/benchmarks/defects4j_c1.json"
    c1_exclude = set()
    try:
        c1_data = json.load(open(c1_path))
        for key in c1_data.get("data", {}).keys():
            m = re.match(r'^(\w+)_(\d+)', key)
            if m:
                c1_exclude.add(f'{m.group(1)}-{m.group(2)}')
    except FileNotFoundError:
        print(f"Warning: {c1_path} not found, no projects excluded")

    bugs = []
    build_ranges = {
        "Chart":   (1, 26,  []),
        "Closure": (1, 133, [63, 93]),
        "Lang":    (1, 65,  [2]),
        "Math":    (1, 106, []),
        "Mockito": (1, 38,  []),
        "Time":    (1, 27,  []),
    }

    for project, (start, end, skips) in build_ranges.items():
        for i in range(start, end + 1):
            if i in skips:
                continue
            bug_id = f"{project}-{i}"
            if bug_id not in c1_exclude:
                bugs.append(bug_id)

    return bugs