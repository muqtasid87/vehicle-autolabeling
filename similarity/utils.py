# File: similarity/utils.py
import difflib
from .configs import CATEGORY_HIERARCHY, CATEGORY_SIMILARITY_GROUPS, SUBCATEGORY_SIMILARITY_GROUPS

def get_category_from_subcategory(subcategory: str) -> str:
    for category, subcategories in CATEGORY_HIERARCHY.items():
        if subcategory in subcategories:
            return category
    return "Unclassified"

def calculate_category_similarity(cat1: str, cat2: str) -> float:
    if cat1 == cat2:
        return 1.0
    pair1 = (cat1, cat2)
    pair2 = (cat2, cat1)
    if pair1 in CATEGORY_SIMILARITY_GROUPS:
        return CATEGORY_SIMILARITY_GROUPS[pair1]
    elif pair2 in CATEGORY_SIMILARITY_GROUPS:
        return CATEGORY_SIMILARITY_GROUPS[pair2]
    return 0.0

def calculate_subcategory_similarity(sub1: str, sub2: str) -> float:
    if sub1 == sub2:
        return 1.0
    cat1 = get_category_from_subcategory(sub1)
    cat2 = get_category_from_subcategory(sub2)
    if cat1 == cat2 and cat1 != "Unclassified":
        pair1 = (sub1, sub2)
        pair2 = (sub2, sub1)
        if pair1 in SUBCATEGORY_SIMILARITY_GROUPS:
            return SUBCATEGORY_SIMILARITY_GROUPS[pair1]
        elif pair2 in SUBCATEGORY_SIMILARITY_GROUPS:
            return SUBCATEGORY_SIMILARITY_GROUPS[pair2]
        else:
            return 0.5
    category_sim = calculate_category_similarity(cat1, cat2)
    return category_sim * 0.6

def string_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def str_sim(s1: str, s2: str) -> float:
    s1 = s1.strip().lower() if s1 else ""
    s2 = s2.strip().lower() if s2 else ""
    if s1 == s2:
        return 1.0
    elif not s1 or not s2:
        return 0.0
    else:
        return string_similarity(s1, s2)

def int_sim(i1: int, i2: int, max_diff: float = 10.0) -> float:
    if i1 == i2:
        return 1.0
    diff = abs(i1 - i2)
    return max(0.0, 1.0 - (diff / max_diff))

def bool_sim(b1: bool, b2: bool) -> float:
    return 1.0 if b1 == b2 else 0.0