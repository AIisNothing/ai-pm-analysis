#!/usr/bin/env python3
"""
AI/大模型产品岗位深度分析脚本（方法示例）
从 JD 样本中提取结构化特征，做自然聚类和高薪差异分析。

注意：本脚本不包含原始数据。如需运行，请使用合法合规且已脱敏的招聘数据，
并将数据库路径配置为环境变量 JD_DB_PATH 或修改下方 DB_PATH。
"""

import sqlite3
import json
import re
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

DB_PATH = os.environ.get("JD_DB_PATH", "sample_jobs.db")

# ============================================================
# 1. 薪资解析
# ============================================================

def parse_salary(s: str):
    """解析薪资描述，返回 (min_k, max_k, months, mid_k)"""
    if not s:
        return None
    s = s.replace("·", "·")
    m = re.match(r"(\d+)-(\d+)K", s)
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    months = 12
    mm = re.search(r"(\d+)薪", s)
    if mm:
        months = int(mm.group(1))
    mid = (lo + hi) / 2
    return {"min": lo, "max": hi, "months": months, "mid": mid}


# ============================================================
# 2. JD 结构化特征提取 — 关键词规则
# ============================================================

# --- 2a. AI 技术信号 ---
AI_TECH_SIGNALS = {
    "大模型/LLM": [r"大模型", r"llm", r"大语言模型", r"foundation\s*model"],
    "Agent/智能体": [r"agent", r"智能体", r"ai\s*agent"],
    "RAG/检索增强": [r"rag", r"检索增强", r"retrieval", r"知识库", r"向量"],
    "Prompt工程": [r"prompt", r"提示词", r"指令工程", r"prompt\s*engineer"],
    "多模态": [r"多模态", r"multi.?modal", r"视觉.*语言", r"图文"],
    "NLP/语义": [r"nlp", r"自然语言处理", r"语义理解", r"文本"],
    "对话/Chatbot": [r"对话", r"chatbot", r"chat\s*bot", r"问答", r"客服机器人"],
    "AIGC/内容生成": [r"aigc", r"内容生成", r"ai\s*生成", r"文生图", r"生成式"],
    "推荐/搜索策略": [r"推荐", r"搜索", r"recall", r"排序", r"ctr", r"策略"],
    "CV/视觉": [r"计算机视觉", r"cv\b", r"图像识别", r"目标检测", r"ocr"],
    "语音/ASR/TTS": [r"语音", r"asr", r"tts", r"speech"],
    "数据标注/训练": [r"标注", r"fine.?tun", r"微调", r"训练数据", r"rlhf", r"sft"],
    "模型评估/效果": [r"模型评估", r"评测", r"benchmark", r"效果评估", r"badcase"],
    "RPA/自动化": [r"rpa", r"自动化", r"workflow", r"工作流"],
    "知识图谱": [r"知识图谱", r"knowledge\s*graph", r"kg\b"],
    "AI基础设施": [r"模型部署", r"推理优化", r"inference", r"gpu", r"算力", r"mlops"],
}

# --- 2b. 产品能力要求 ---
PRODUCT_CAPABILITY_SIGNALS = {
    "需求分析与PRD": [r"需求分析", r"prd", r"需求文档", r"产品需求", r"功能设计"],
    "产品规划/路线图": [r"产品规划", r"roadmap", r"路线图", r"版本规划", r"产品战略"],
    "用户研究": [r"用户研究", r"用户调研", r"用户访谈", r"用户画像", r"用户需求"],
    "数据分析/指标": [r"数据分析", r"数据驱动", r"指标", r"漏斗", r"转化率", r"留存"],
    "AB实验": [r"a/?b\s*(test|实验|测试)", r"实验设计", r"对照实验"],
    "竞品分析": [r"竞品", r"竞争分析", r"市场调研", r"行业分析"],
    "项目管理/推进": [r"项目管理", r"项目推进", r"跨.*协同", r"推动.*落地", r"项目落地"],
    "业务闭环/ROI": [r"roi\b", r"业务闭环", r"商业价值", r"业务价值", r"营收", r"商业模式", r"盈利"],
    "原型/交互设计": [r"原型", r"交互设计", r"wireframe", r"figma", r"axure"],
    "用户增长": [r"用户增长", r"增长", r"拉新", r"获客", r"growth"],
    "商业化/变现": [r"商业化", r"变现", r"monetiz", r"付费", r"广告.*变现"],
    "平台化/架构": [r"平台化", r"中台", r"能力平台", r"架构设计", r"系统设计", r"平台产品"],
    "场景抽象/方案设计": [r"场景.*(?:抽象|梳理|分析|设计)", r"解决方案", r"产品方案", r"落地方案"],
    "技术理解/沟通": [r"技术.*(?:理解|背景|沟通|方案)", r"算法.*(?:理解|沟通)", r"研发.*(?:协同|沟通|对接)"],
    "从0到1": [r"从0到1", r"0.?1", r"从零到一", r"从无到有", r"开创"],
    "团队管理": [r"团队管理", r"带.*团队", r"管理.*团队", r"leader", r"负责人"],
}

# --- 2c. 产品服务对象 ---
SERVICE_OBJECT_SIGNALS = {
    "C端用户": [r"c\s*端", r"to\s*c", r"用户端", r"消费者", r"个人用户", r"app"],
    "B端企业": [r"b\s*端", r"to\s*b", r"企业客户", r"企业用户", r"saas", r"crm"],
    "内部业务": [r"内部.*(?:业务|系统|工具|平台)", r"中后台", r"效率工具", r"内部提效"],
    "开发者": [r"开发者", r"developer", r"api", r"sdk", r"开放平台"],
    "政府/G端": [r"g\s*端", r"政务", r"政府", r"公共"],
}

# --- 2d. 业务场景 ---
BUSINESS_SCENE_SIGNALS = {
    "电商": [r"电商", r"商品", r"购物", r"交易", r"订单"],
    "金融": [r"金融", r"银行", r"保险", r"证券", r"风控", r"信贷"],
    "教育": [r"教育", r"教学", r"学习", r"培训", r"课程"],
    "医疗": [r"医疗", r"健康", r"医院", r"医学", r"诊断"],
    "出行/汽车": [r"出行", r"汽车", r"自动驾驶", r"车联网", r"座舱"],
    "内容/社区": [r"内容", r"社区", r"ugc", r"短视频", r"直播"],
    "办公/协作": [r"办公", r"协作", r"文档", r"会议", r"im\b"],
    "营销/广告": [r"营销", r"广告", r"投放", r"创意", r"素材"],
    "客服/服务": [r"客服", r"售后", r"工单", r"服务.*平台"],
    "人力/招聘": [r"人力", r"hr\b", r"招聘", r"简历"],
    "安全": [r"安全", r"风控", r"反欺诈", r"合规"],
    "搜索": [r"搜索", r"搜索引擎", r"信息检索"],
    "游戏": [r"游戏", r"npc", r"游戏.*ai"],
    "IoT/硬件": [r"iot", r"硬件", r"智能家居", r"设备", r"穿戴"],
    "通用/企业服务": [r"企业服务", r"数字化", r"信息化"],
}

# --- 2e. 岗位成熟度/阶段性信号 ---
MATURITY_SIGNALS = {
    "探索期/从0到1": [r"从0到1", r"0.?1", r"从零", r"探索", r"孵化", r"创新.*业务", r"新方向"],
    "成长期/快速迭代": [r"快速迭代", r"敏捷", r"小步快跑", r"迭代优化"],
    "成熟期/规模化": [r"规模化", r"平台化", r"标准化", r"体系化", r"系统性"],
}

# --- 2f. 高阶信号 ---
SENIOR_SIGNALS = {
    "产品负责人/owner": [r"产品负责人", r"产品.*owner", r"负责.*产品线", r"负责.*产品方向", r"独立负责"],
    "战略/全局": [r"战略", r"全局", r"布局", r"规划.*方向", r"定义.*方向"],
    "团队管理": [r"管理.*团队", r"带.*团队", r"团队.*管理", r"leader", r"(?:组建|搭建).*团队"],
    "跨部门/跨BU": [r"跨.*部门", r"跨.*团队", r"跨.*bu\b", r"跨.*业务", r"协调.*资源"],
    "方法论/体系": [r"方法论", r"体系", r"流程.*建设", r"标准.*制定", r"规范"],
}


def extract_signals(text: str, signal_dict: dict) -> dict:
    """从文本中提取信号，返回 {signal_name: True/False}"""
    text_lower = text.lower()
    result = {}
    for name, patterns in signal_dict.items():
        found = False
        for pat in patterns:
            if re.search(pat, text_lower):
                found = True
                break
        result[name] = found
    return result


def extract_all_features(row) -> dict:
    """从一条记录中提取所有结构化特征

    期望的数据表字段：
      id, title, company, salary_desc, experience, education,
      area, company_industry, company_scale, company_stage,
      jd_text, skills_json, direction
    """
    jd = row["jd_text"] or ""
    title = row["title"] or ""
    skills_raw = row["skills_json"] or "[]"
    try:
        skills = json.loads(skills_raw)
    except:
        skills = []

    # 合并文本用于匹配
    full_text = f"{title} {' '.join(skills)} {jd}"

    # 薪资
    sal = parse_salary(row["salary_desc"])

    features = {
        "id": row["id"],
        "title": title,
        "company": row["company"],
        "salary_desc": row["salary_desc"],
        "salary_mid": sal["mid"] if sal else None,
        "salary_min": sal["min"] if sal else None,
        "salary_max": sal["max"] if sal else None,
        "salary_months": sal["months"] if sal else None,
        "experience": row["experience"],
        "education": row["education"],
        "area": row.get("area", ""),
        "company_industry": row.get("company_industry", ""),
        "company_scale": row.get("company_scale", ""),
        "company_stage": row.get("company_stage", ""),
        "jd_length": len(jd),
        "skills": skills,
    }

    # 提取各类信号
    features["ai_tech"] = extract_signals(full_text, AI_TECH_SIGNALS)
    features["product_cap"] = extract_signals(full_text, PRODUCT_CAPABILITY_SIGNALS)
    features["service_obj"] = extract_signals(full_text, SERVICE_OBJECT_SIGNALS)
    features["business_scene"] = extract_signals(full_text, BUSINESS_SCENE_SIGNALS)
    features["maturity"] = extract_signals(full_text, MATURITY_SIGNALS)
    features["senior"] = extract_signals(full_text, SENIOR_SIGNALS)

    return features


# ============================================================
# 3. 数据加载和筛选
# ============================================================

def load_data():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM jd_records
        WHERE direction='AI/大模型产品'
    """).fetchall()
    conn.close()
    return rows


def filter_analysis_set(features_list: list) -> list:
    """按分析口径筛选：3-10年经验，本科+，salary_max >= 30K，排除 salary_mid >= 100K"""
    valid_exp = {"3-5年", "5-10年"}
    valid_edu = {"本科", "硕士", "博士"}

    filtered = []
    for f in features_list:
        if f["experience"] not in valid_exp:
            continue
        if f["education"] not in valid_edu:
            continue
        if f["salary_max"] is None or f["salary_max"] < 30:
            continue
        if f["salary_mid"] is not None and f["salary_mid"] >= 100:
            continue
        filtered.append(f)
    return filtered


# ============================================================
# 4. 统计工具函数
# ============================================================

def percentile(values, p):
    """计算百分位数"""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def signal_rate(features_list, signal_category, signal_name):
    """计算某个信号在列表中的出现率"""
    if not features_list:
        return 0
    count = sum(1 for f in features_list if f[signal_category].get(signal_name, False))
    return count / len(features_list)


def signal_counts(features_list, signal_category):
    """统计某类信号的所有出现次数"""
    c = Counter()
    for f in features_list:
        for name, val in f[signal_category].items():
            if val:
                c[name] += 1
    return c


# ============================================================
# 5. 主分析逻辑
# ============================================================

def analyze_overall_profile(all_feats, filtered_feats):
    """第一章：整体人才画像"""
    print("=" * 70)
    print("第一章：AI 产品方向整体人才画像")
    print("=" * 70)

    n_all = len(all_feats)
    n_filt = len(filtered_feats)

    print(f"\n样本概况：")
    print(f"  全量 AI 方向记录：{n_all} 条")
    print(f"  筛选后（3-10年+本科+salary_max≥30K）：{n_filt} 条")

    # 使用全量数据做整体画像
    target = all_feats
    n = len(target)

    # 薪资
    mids = [f["salary_mid"] for f in target if f["salary_mid"]]
    mids.sort()
    print(f"\n--- 薪资分布（全量 {len(mids)} 条） ---")
    print(f"  P25={percentile(mids,25):.1f}K  P50={percentile(mids,50):.1f}K  P75={percentile(mids,75):.1f}K  P90={percentile(mids,90):.1f}K")

    # 筛选后薪资
    filt_mids = [f["salary_mid"] for f in filtered_feats if f["salary_mid"]]
    filt_mids.sort()
    p75 = percentile(filt_mids, 75)
    print(f"\n--- 薪资分布（筛选后 {len(filt_mids)} 条） ---")
    print(f"  P25={percentile(filt_mids,25):.1f}K  P50={percentile(filt_mids,50):.1f}K  P75={p75:.1f}K  P90={percentile(filt_mids,90):.1f}K")

    # AI 技术信号
    print(f"\n--- AI 技术信号（全量 {n} 条） ---")
    ai_counts = signal_counts(target, "ai_tech")
    for name, cnt in ai_counts.most_common():
        print(f"  {name}: {cnt} ({cnt/n*100:.1f}%)")

    # 产品能力要求
    print(f"\n--- 产品能力要求（全量 {n} 条） ---")
    cap_counts = signal_counts(target, "product_cap")
    for name, cnt in cap_counts.most_common():
        print(f"  {name}: {cnt} ({cnt/n*100:.1f}%)")

    # 服务对象
    print(f"\n--- 产品服务对象（全量 {n} 条） ---")
    obj_counts = signal_counts(target, "service_obj")
    for name, cnt in obj_counts.most_common():
        print(f"  {name}: {cnt} ({cnt/n*100:.1f}%)")

    # 业务场景
    print(f"\n--- 业务场景（全量 {n} 条） ---")
    scene_counts = signal_counts(target, "business_scene")
    for name, cnt in scene_counts.most_common():
        print(f"  {name}: {cnt} ({cnt/n*100:.1f}%)")

    # 高阶信号
    print(f"\n--- 高阶信号（全量 {n} 条） ---")
    senior_counts = signal_counts(target, "senior")
    for name, cnt in senior_counts.most_common():
        print(f"  {name}: {cnt} ({cnt/n*100:.1f}%)")

    # 岗位成熟度
    print(f"\n--- 岗位成熟度（全量 {n} 条） ---")
    mat_counts = signal_counts(target, "maturity")
    for name, cnt in mat_counts.most_common():
        print(f"  {name}: {cnt} ({cnt/n*100:.1f}%)")

    return p75


def analyze_high_salary_diff(filtered_feats, p75):
    """第四章：高薪 vs 普通 JD 差异"""
    print("\n" + "=" * 70)
    print("第四章：高薪 vs 普通岗位 JD 差异分析")
    print("=" * 70)

    high = [f for f in filtered_feats if f["salary_mid"] and f["salary_mid"] >= p75]
    normal = [f for f in filtered_feats if f["salary_mid"] and f["salary_mid"] < p75]

    print(f"\n高薪组（salary_mid >= {p75:.1f}K）：{len(high)} 条")
    print(f"普通组（salary_mid < {p75:.1f}K）：{len(normal)} 条")

    if not high or not normal:
        print("样本不足，无法比较")
        return

    # 高薪组薪资
    h_mids = sorted([f["salary_mid"] for f in high if f["salary_mid"]])
    n_mids = sorted([f["salary_mid"] for f in normal if f["salary_mid"]])
    print(f"高薪组薪资中位数：{percentile(h_mids,50):.1f}K")
    print(f"普通组薪资中位数：{percentile(n_mids,50):.1f}K")

    # 对比所有信号类别
    for cat_name, cat_key in [
        ("AI技术信号", "ai_tech"),
        ("产品能力要求", "product_cap"),
        ("服务对象", "service_obj"),
        ("业务场景", "business_scene"),
        ("高阶信号", "senior"),
        ("岗位成熟度", "maturity"),
    ]:
        print(f"\n--- {cat_name} 差异 ---")
        print(f"{'信号':<25} {'高薪率':>8} {'普通率':>8} {'差异':>8} {'高薪N':>6} {'普通N':>6}")

        all_signals = set()
        for f in filtered_feats:
            all_signals.update(f[cat_key].keys())

        diffs = []
        for sig in all_signals:
            h_rate = signal_rate(high, cat_key, sig)
            n_rate = signal_rate(normal, cat_key, sig)
            h_count = sum(1 for f in high if f[cat_key].get(sig, False))
            n_count = sum(1 for f in normal if f[cat_key].get(sig, False))
            if h_rate > 0 or n_rate > 0:
                ratio = h_rate / n_rate if n_rate > 0 else (99.0 if h_rate > 0 else 0)
                diffs.append((sig, h_rate, n_rate, ratio, h_count, n_count))

        # 按差异倍数排序
        diffs.sort(key=lambda x: -x[3])
        for sig, h_rate, n_rate, ratio, h_c, n_c in diffs:
            if h_rate > 0.01 or n_rate > 0.01:
                caution = " ⚠️" if h_c < 20 and n_c < 20 else ""
                print(f"  {sig:<23} {h_rate:>7.1%} {n_rate:>7.1%} {ratio:>7.2f}x {h_c:>5} {n_c:>5}{caution}")

    return high, normal


def natural_clustering(filtered_feats, p75):
    """第三章：基于数据自然归纳岗位群"""
    print("\n" + "=" * 70)
    print("第三章：AI 产品岗位自然聚类")
    print("=" * 70)

    # 基于 JD 的多维特征做规则聚类
    # 先统计高频组合模式，然后归纳

    clusters = defaultdict(list)

    for f in filtered_feats:
        title_lower = f["title"].lower()
        jd_lower = ""  # 不重新读 JD，用提取的特征

        ai_techs = [k for k, v in f["ai_tech"].items() if v]
        caps = [k for k, v in f["product_cap"].items() if v]
        scenes = [k for k, v in f["business_scene"].items() if v]
        objs = [k for k, v in f["service_obj"].items() if v]
        senior = [k for k, v in f["senior"].items() if v]

        # 聚类逻辑：按核心职责 + 服务对象 + 技术信号综合判断
        cluster = None

        # 1. Agent/智能体产品
        if f["ai_tech"].get("Agent/智能体") and ("agent" in title_lower or "智能体" in title_lower):
            cluster = "Agent/智能体产品"

        # 2. 对话/Chatbot 产品（明确做对话类产品）
        elif f["ai_tech"].get("对话/Chatbot") and not cluster:
            if any(kw in title_lower for kw in ["对话", "chat", "客服", "问答"]):
                cluster = "对话/Chatbot产品"
            elif f["business_scene"].get("客服/服务"):
                cluster = "对话/Chatbot产品"

        # 3. AIGC/内容生成产品
        if not cluster and f["ai_tech"].get("AIGC/内容生成"):
            if any(kw in title_lower for kw in ["aigc", "生成", "内容", "创作"]):
                cluster = "AIGC/内容生成产品"

        # 4. 大模型平台/基础设施
        if not cluster:
            if f["ai_tech"].get("AI基础设施") or f["ai_tech"].get("数据标注/训练"):
                if f["product_cap"].get("平台化/架构") or "平台" in title_lower or "基础" in title_lower:
                    cluster = "大模型平台/基础设施"
            if not cluster and f["ai_tech"].get("大模型/LLM") and f["product_cap"].get("平台化/架构"):
                cluster = "大模型平台/基础设施"

        # 5. 推荐/搜索策略产品
        if not cluster and f["ai_tech"].get("推荐/搜索策略"):
            if any(kw in title_lower for kw in ["策略", "推荐", "搜索", "算法"]):
                cluster = "推荐/搜索/策略产品"

        # 6. 多模态/CV/语音产品
        if not cluster:
            if f["ai_tech"].get("多模态") or f["ai_tech"].get("CV/视觉") or f["ai_tech"].get("语音/ASR/TTS"):
                if any(kw in title_lower for kw in ["多模态", "视觉", "语音", "图像"]):
                    cluster = "多模态/视觉/语音产品"

        # 7. 商业化/营销 AI 产品
        if not cluster:
            if f["product_cap"].get("商业化/变现") or f["business_scene"].get("营销/广告"):
                if any(kw in title_lower for kw in ["商业", "营销", "广告", "增长"]):
                    cluster = "AI商业化/营销产品"
            elif f["product_cap"].get("业务闭环/ROI") and f["product_cap"].get("商业化/变现"):
                cluster = "AI商业化/营销产品"

        # 8. 行业 AI 产品（金融/教育/医疗/汽车等）
        if not cluster:
            industry_scenes = ["金融", "教育", "医疗", "出行/汽车"]
            matched_scenes = [s for s in industry_scenes if f["business_scene"].get(s)]
            if matched_scenes:
                cluster = "行业AI产品"

        # 9. B端/企业服务 AI 产品
        if not cluster:
            if f["service_obj"].get("B端企业") and not f["service_obj"].get("C端用户"):
                cluster = "B端/企业AI产品"

        # 10. 内部提效/工具类 AI 产品
        if not cluster:
            if f["service_obj"].get("内部业务"):
                cluster = "内部提效/AI工具"

        # 11. 大模型应用产品（泛 LLM 应用，不属于上面任何类）
        if not cluster:
            if f["ai_tech"].get("大模型/LLM") or f["ai_tech"].get("RAG/检索增强") or f["ai_tech"].get("Prompt工程"):
                cluster = "大模型应用产品(泛)"

        # 12. C端 AI 产品
        if not cluster:
            if f["service_obj"].get("C端用户"):
                cluster = "C端AI产品"

        # 兜底
        if not cluster:
            cluster = "综合/其他AI产品"

        clusters[cluster].append(f)

    # 输出每个聚类的统计
    for cluster_name in sorted(clusters.keys(), key=lambda x: -len(clusters[x])):
        members = clusters[cluster_name]
        n = len(members)
        if n < 5:
            # 太少的合并到"综合"
            clusters["综合/其他AI产品"].extend(members)
            continue

        mids = sorted([f["salary_mid"] for f in members if f["salary_mid"]])
        high_count = sum(1 for f in members if f["salary_mid"] and f["salary_mid"] >= p75)

        print(f"\n{'─' * 60}")
        print(f"📌 {cluster_name}  (n={n}, 占比 {n/len(filtered_feats)*100:.1f}%)")
        print(f"{'─' * 60}")

        if mids:
            print(f"  薪资：P50={percentile(mids,50):.1f}K  P75={percentile(mids,75):.1f}K  P90={percentile(mids,90):.1f}K")
            print(f"  高薪占比（>={p75:.0f}K）：{high_count}/{n} = {high_count/n*100:.1f}%")

        # 经验分布
        exp_c = Counter(f["experience"] for f in members)
        print(f"  经验分布：{dict(exp_c.most_common(5))}")

        # AI 技术信号 TOP5
        ai_c = signal_counts(members, "ai_tech")
        top_ai = [(name, cnt) for name, cnt in ai_c.most_common(8) if cnt >= 3]
        if top_ai:
            print(f"  高频AI技术信号：")
            for name, cnt in top_ai:
                print(f"    {name}: {cnt/n*100:.0f}%")

        # 产品能力 TOP5
        cap_c = signal_counts(members, "product_cap")
        top_cap = [(name, cnt) for name, cnt in cap_c.most_common(8) if cnt >= 3]
        if top_cap:
            print(f"  高频能力要求：")
            for name, cnt in top_cap:
                print(f"    {name}: {cnt/n*100:.0f}%")

        # 业务场景
        scene_c = signal_counts(members, "business_scene")
        top_scene = [(name, cnt) for name, cnt in scene_c.most_common(5) if cnt >= 3]
        if top_scene:
            print(f"  业务场景：")
            for name, cnt in top_scene:
                print(f"    {name}: {cnt/n*100:.0f}%")

        # 高阶信号
        senior_c = signal_counts(members, "senior")
        top_senior = [(name, cnt) for name, cnt in senior_c.most_common(5) if cnt >= 3]
        if top_senior:
            print(f"  高阶信号：")
            for name, cnt in top_senior:
                print(f"    {name}: {cnt/n*100:.0f}%")

        # 样本可信度
        if n >= 100:
            print(f"  样本可信度：A级（≥100）")
        elif n >= 50:
            print(f"  样本可信度：B级（50-99）")
        elif n >= 20:
            print(f"  样本可信度：C级（20-49）")
        else:
            print(f"  样本可信度：D级（<20）⚠️ 结论仅供参考")

        # 典型标题
        title_c = Counter(f["title"] for f in members)
        top_titles = title_c.most_common(5)
        print(f"  典型岗位：{', '.join(t for t, _ in top_titles)}")

    return clusters


def capability_factor_analysis(filtered_feats, high, normal, p75):
    """第五章：能力因子分析"""
    print("\n" + "=" * 70)
    print("第五章：高薪相关能力因子分析")
    print("=" * 70)

    print(f"\n高薪定义：salary_mid >= {p75:.1f}K")
    print(f"高薪组：{len(high)} 条  普通组：{len(normal)} 条")

    # 汇总所有信号
    all_signals = []

    for cat_key, cat_label in [
        ("ai_tech", "AI技术"),
        ("product_cap", "产品能力"),
        ("senior", "高阶"),
        ("maturity", "成熟度"),
    ]:
        for sig_name in set().union(*[set(f[cat_key].keys()) for f in filtered_feats]):
            h_rate = signal_rate(high, cat_key, sig_name)
            n_rate = signal_rate(normal, cat_key, sig_name)
            all_rate = signal_rate(filtered_feats, cat_key, sig_name)
            h_count = sum(1 for f in high if f[cat_key].get(sig_name, False))
            n_count = sum(1 for f in normal if f[cat_key].get(sig_name, False))

            if all_rate < 0.02:  # 总出现率 < 2% 跳过
                continue

            ratio = h_rate / n_rate if n_rate > 0.005 else (99 if h_rate > 0 else 0)
            diff = h_rate - n_rate

            # 分类：门槛 / 加分 / 高薪区分
            if all_rate >= 0.30:
                role = "门槛能力"
            elif ratio >= 1.5 and h_rate >= 0.15:
                role = "高薪区分能力"
            elif ratio >= 1.3 and h_rate >= 0.10:
                role = "加分能力"
            elif ratio <= 0.7 and n_rate >= 0.10:
                role = "普通岗高频"
            else:
                role = "中性"

            all_signals.append({
                "category": cat_label,
                "name": sig_name,
                "all_rate": all_rate,
                "h_rate": h_rate,
                "n_rate": n_rate,
                "ratio": ratio,
                "diff": diff,
                "h_count": h_count,
                "n_count": n_count,
                "role": role,
            })

    # 按差异排序输出
    all_signals.sort(key=lambda x: -x["diff"])

    print(f"\n{'类别':<8} {'能力因子':<25} {'全样本':>7} {'高薪组':>7} {'普通组':>7} {'倍数':>6} {'角色':<12} {'高薪N':>5}")
    print("-" * 100)
    for s in all_signals:
        caution = " ⚠️" if s["h_count"] < 15 else ""
        print(f"  {s['category']:<6} {s['name']:<23} {s['all_rate']:>6.1%} {s['h_rate']:>6.1%} {s['n_rate']:>6.1%} {s['ratio']:>5.2f}x {s['role']:<10} {s['h_count']:>4}{caution}")

    return all_signals


def cross_analysis(clusters, filtered_feats, p75):
    """第六章：技术信号、能力因子、岗位群的关联分析"""
    print("\n" + "=" * 70)
    print("第六章：技术信号 × 能力因子 × 岗位群 关联分析")
    print("=" * 70)

    # 技术信号 × 岗位群 热力图
    print(f"\n--- AI技术信号 × 岗位群 ---")
    cluster_names = sorted(clusters.keys(), key=lambda x: -len(clusters[x]))
    cluster_names = [c for c in cluster_names if len(clusters[c]) >= 10]

    all_ai_signals = list(AI_TECH_SIGNALS.keys())

    print(f"\n{'信号':<20}", end="")
    for cn in cluster_names:
        short = cn[:10]
        print(f" {short:>10}", end="")
    print()

    for sig in all_ai_signals:
        rates = []
        for cn in cluster_names:
            members = clusters[cn]
            rate = signal_rate(members, "ai_tech", sig)
            rates.append(rate)
        if max(rates) < 0.05:
            continue
        print(f"  {sig:<18}", end="")
        for r in rates:
            if r >= 0.3:
                mark = f"{'█':>10}"
            elif r >= 0.15:
                mark = f"{'▓':>10}"
            elif r >= 0.05:
                mark = f"{'░':>10}"
            else:
                mark = f"{'·':>10}"
            print(f" {r:>9.0%}", end="")
        print()

    # 能力因子 × 岗位群
    print(f"\n--- 产品能力 × 岗位群 ---")
    all_cap_signals = list(PRODUCT_CAPABILITY_SIGNALS.keys())

    print(f"\n{'能力':<20}", end="")
    for cn in cluster_names:
        short = cn[:10]
        print(f" {short:>10}", end="")
    print()

    for sig in all_cap_signals:
        rates = []
        for cn in cluster_names:
            members = clusters[cn]
            rate = signal_rate(members, "product_cap", sig)
            rates.append(rate)
        if max(rates) < 0.05:
            continue
        print(f"  {sig:<18}", end="")
        for r in rates:
            print(f" {r:>9.0%}", end="")
        print()

    # 经验 × 高薪率
    print(f"\n--- 经验 × 高薪占比 ---")
    for exp in ["3-5年", "5-10年"]:
        exp_feats = [f for f in filtered_feats if f["experience"] == exp]
        exp_high = sum(1 for f in exp_feats if f["salary_mid"] and f["salary_mid"] >= p75)
        n_exp = len(exp_feats)
        if n_exp > 0:
            print(f"  {exp}: {exp_high}/{n_exp} = {exp_high/n_exp*100:.1f}% 高薪 (n={n_exp})")


# ============================================================
# 6. 主函数
# ============================================================

def main():
    print("加载数据...")
    rows = load_data()
    print(f"AI方向总记录：{len(rows)} 条")

    print("提取结构化特征...")
    all_feats = [extract_all_features(r) for r in rows]

    print("按分析口径筛选...")
    filtered = filter_analysis_set(all_feats)
    print(f"筛选后：{len(filtered)} 条")

    # 第一章
    p75 = analyze_overall_profile(all_feats, filtered)

    # 第三章：聚类
    clusters = natural_clustering(filtered, p75)

    # 第四章：高薪差异
    result = analyze_high_salary_diff(filtered, p75)
    if result:
        high, normal = result
    else:
        return

    # 第五章：能力因子
    capability_factor_analysis(filtered, high, normal, p75)

    # 第六章：关联分析
    cross_analysis(clusters, filtered, p75)

    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
