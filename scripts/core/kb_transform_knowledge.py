"""
kb_transform_knowledge.py
Transform knowledge.jsonl files to canonical KB format with versioning and scope.

Usage:
  # Basic transform
  python scripts/core/kb_transform_knowledge.py --input data_kb/raw/knowledge_wms.jsonl --domain wms

  # With version and source info
  python scripts/core/kb_transform_knowledge.py \
      --input data_kb/raw/knowledge_wms.jsonl \
      --domain wms \
      --source-type video_workshop \
      --version 2025.1

  # Append to existing canonical (instead of overwrite)
  python scripts/core/kb_transform_knowledge.py \
      --input data_kb/raw/knowledge_wms_session2.jsonl \
      --domain wms \
      --append
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# SCOPE CLASSIFICATION
# =============================================================================

# Platform-level keywords (shared across all products)
PLATFORM_KEYWORDS = [
    # Security & Auth
    "sso", "single sign-on", "authentication", "authorization", "rbac",
    "security", "encryption", "compliance", "soc2", "iso 27001",
    # Infrastructure
    "azure", "aws", "cloud", "data center", "region", "availability zone",
    "disaster recovery", "failover", "backup", "rto", "rpo",
    # Platform Services
    "platform data cloud", "snowflake", "data share", "api management",
    "api catalog", "rest api", "openapi", "oas",
    # SLAs
    "sla", "uptime", "99.7", "99.95", "availability", "incident", "severity",
    "tam", "technical account manager",
    # General Platform
    "saas", "multi-tenant", "single-tenant", "subscription",
]

# Product-specific keywords per domain
DOMAIN_KEYWORDS = {
    "wms": [
        "warehouse", "inventory", "picking", "putaway", "receiving",
        "labor", "workforce", "task", "wave", "rf", "android app",
        "wms", "warehouse management", "yard", "dock", "bin", "location",
        "pallet", "carton", "sku", "lpn", "asn", "shipment",
        "automation", "conveyor", "sorter", "robotics",
        "health console", "warehouse health",
    ],
    "planning": [
        "demand", "forecast", "supply", "planning", "s&op", "ibp",
        "production", "scheduling", "mrp", "inventory optimization",
        "demand sensing", "machine learning forecast",
    ],
    "catman": [
        "category", "assortment", "space", "planogram", "shelf",
        "merchandising", "retail", "store", "sku rationalization",
    ],
    "logistics": [
        "transportation", "tms", "freight", "carrier", "route",
        "shipment", "load", "trailer", "fleet", "delivery",
    ],
}


def classify_scope(question: str, answer: str, domain: str) -> Tuple[str, float]:
    """
    Classify if entry is platform-level or product-specific.
    
    Returns:
        tuple: (scope, confidence)
        - scope: "platform" | "product_specific"
        - confidence: 0.0 to 1.0
    """
    text = (question + " " + answer).lower()
    
    # Count platform keywords
    platform_score = sum(1 for kw in PLATFORM_KEYWORDS if kw in text)
    
    # Count domain-specific keywords
    domain_kws = DOMAIN_KEYWORDS.get(domain, [])
    domain_score = sum(1 for kw in domain_kws if kw in text)
    
    total = platform_score + domain_score
    
    if total == 0:
        return "product_specific", 0.5  # Default to product-specific with low confidence
    
    if domain_score > platform_score:
        confidence = domain_score / total
        return "product_specific", round(confidence, 2)
    elif platform_score > domain_score:
        confidence = platform_score / total
        return "platform", round(confidence, 2)
    else:
        # Equal - default to product_specific
        return "product_specific", 0.5


# =============================================================================
# CATEGORY CLASSIFICATION
# =============================================================================

CATEGORY_RULES = [
    # (keywords, category, subcategory)
    (["sla", "uptime", "99.7", "99.95", "availability"], "SLA & Availability", "Service Level Agreements"),
    (["disaster", "recovery", "rto", "rpo", "failover", "backup"], "SLA & Availability", "Disaster Recovery"),
    (["incident", "severity", "support", "24x7"], "Support", "Incident Management"),
    (["tam", "technical account manager"], "Support", "Technical Account Management"),
    (["api", "rest", "integration", "openapi", "oas"], "Integration", "APIs"),
    (["cig", "cognitive integration", "mapping"], "Integration", "Cognitive Integration"),
    (["security", "sso", "authentication", "encryption", "rbac"], "Security & Compliance", "Authentication"),
    (["azure", "region", "data center", "availability zone"], "Infrastructure", "Cloud Infrastructure"),
    (["microservice", "container", "kubernetes"], "Infrastructure", "Architecture"),
    (["extensibility", "extension", "hook", "customiz"], "Extensibility", "Runtime Extensibility"),
    (["workflow", "orchestrat"], "Platform Capabilities", "Workflow Orchestration"),
    (["data function", "sql"], "Platform Capabilities", "Data Functions"),
    (["label", "report", "print"], "Platform Capabilities", "Labels & Reports"),
    (["android", "mobile", "rf", "user interface", "ui"], "User Experience", "Mobile & UI"),
    (["console", "health", "monitoring"], "User Experience", "Monitoring"),
    (["warehouse", "inventory", "picking", "putaway"], "WMS Features", "Warehouse Operations"),
    (["labor", "workforce", "task"], "WMS Features", "Labor Management"),
    (["automation", "conveyor", "robotics"], "WMS Features", "Automation"),
    (["archive", "retention", "historical"], "Data Management", "Archival"),
    (["data as a service", "daas", "snowflake"], "Data Management", "Data as a Service"),
    (["version", "2024", "2025", "2026", "upgrade"], "Versioning", "Release Information"),
    (["configuration", "personalization"], "Configuration", "System Configuration"),
]


def categorize_entry(question: str, answer: str) -> Tuple[str, str]:
    """Determine category and subcategory based on content."""
    text = (question + " " + answer).lower()
    
    for keywords, category, subcategory in CATEGORY_RULES:
        if any(kw in text for kw in keywords):
            return category, subcategory
    
    return "General", "Overview"


# =============================================================================
# KEYWORD EXTRACTION
# =============================================================================

KEYWORD_MAP = {
    "SLA": ["sla", "service level"],
    "Availability": ["availability", "uptime"],
    "Disaster Recovery": ["disaster", "recovery", "failover"],
    "RTO": ["rto", "recovery time"],
    "API": ["api", "rest api"],
    "REST": ["rest"],
    "Integration": ["integration", "integrate"],
    "Security": ["security", "secure"],
    "SSO": ["sso", "single sign"],
    "Azure": ["azure"],
    "SaaS": ["saas"],
    "Extensibility": ["extensibility", "extension"],
    "Hooks": ["hook", "pre-hook", "post-hook"],
    "Workflow": ["workflow"],
    "Data Functions": ["data function"],
    "TAM": ["tam", "technical account"],
    "Support": ["support", "24x7"],
    "WMS": ["wms", "warehouse management"],
    "Inventory": ["inventory"],
    "Labor Management": ["labor"],
    "Android": ["android"],
    "Mobile": ["mobile", "rf device"],
    "Microservices": ["microservice"],
    "CIG": ["cig", "cognitive integration"],
    "Versioning": ["version", "2024", "2025"],
}


def extract_keywords(question: str, answer: str) -> List[str]:
    """Extract relevant keywords from Q&A."""
    text = (question + " " + answer).lower()
    
    found = []
    for keyword, patterns in KEYWORD_MAP.items():
        if any(p in text for p in patterns):
            found.append(keyword)
    
    return found[:10]  # Limit to 10 keywords


# =============================================================================
# VERSION DETECTION
# =============================================================================

def detect_version_info(question: str, answer: str) -> dict:
    """
    Detect version-related information from content.
    
    Returns dict with valid_from, valid_until hints.
    """
    text = (question + " " + answer).lower()
    
    version_info = {
        "valid_from": None,
        "valid_until": None,
        "mentions_versions": [],
        "is_version_specific": False,
    }
    
    # Detect version mentions
    import re
    version_patterns = [
        r"version\s*(\d{4})",
        r"(\d{4})\s*version",
        r"in\s*(\d{4})",
        r"(\d{4})\.(\d)",  # e.g., 2025.1
    ]
    
    for pattern in version_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                version_info["mentions_versions"].append(f"{match[0]}.{match[1]}" if len(match) > 1 else match[0])
            else:
                version_info["mentions_versions"].append(match)
    
    # Clean up
    version_info["mentions_versions"] = list(set(version_info["mentions_versions"]))
    
    # Check for deprecation language
    deprecation_keywords = ["removed", "deprecated", "no longer", "replaced", "obsolete", "not present"]
    if any(kw in text for kw in deprecation_keywords):
        version_info["is_version_specific"] = True
    
    # Check for future language
    future_keywords = ["coming", "will be", "planned", "roadmap", "future"]
    if any(kw in text for kw in future_keywords):
        version_info["is_version_specific"] = True
    
    return version_info


# =============================================================================
# MAIN TRANSFORMER
# =============================================================================

def transform_knowledge(
    input_path: Path,
    output_path: Path,
    domain: str,
    source_type: str = "video_workshop",
    version: str = None,
    append: bool = False,
):
    """
    Transform knowledge.jsonl to canonical format.
    
    Args:
        input_path: Path to input .jsonl file
        output_path: Path to output canonical JSON
        domain: Domain tag (wms, planning, catman, logistics)
        source_type: Source type (video_workshop, document, meeting, etc.)
        version: Product version this applies to (e.g., "2025.1")
        append: If True, append to existing file instead of overwrite
    """
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return

    print(f"[INFO] Loading {input_path.name}...")
    
    # Load input
    entries = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    print(f"   -> {len(entries)} entries")
    
    # Load existing if appending
    existing = []
    next_id = 0
    
    if append and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        # Find highest ID
        for item in existing:
            kb_id = item.get("kb_id", "")
            if kb_id.startswith(f"{domain}_"):
                try:
                    num = int(kb_id.split("_")[1])
                    next_id = max(next_id, num + 1)
                except:
                    pass
        print(f"   -> Appending to {len(existing)} existing entries")
    
    # Transform
    today = datetime.now().strftime("%Y-%m-%d")
    canonical = []
    
    for idx, entry in enumerate(entries):
        question = entry.get("question", "").replace("[REDACTED]", "Blue Yonder")
        answer = entry.get("answer", "").replace("[REDACTED]", "Blue Yonder")
        q_type = entry.get("type", "WHAT")
        frame_id = entry.get("frame_id", entry.get("source_slide"))
        
        if not question or not answer:
            continue
        
        # Classify
        scope, scope_confidence = classify_scope(question, answer, domain)
        category, subcategory = categorize_entry(question, answer)
        keywords = extract_keywords(question, answer)
        version_info = detect_version_info(question, answer)
        
        kb_id = f"{domain}_{next_id + idx:04d}"
        
        # Build search blob
        search_blob = f"DOMAIN: {domain} | SCOPE: {scope} || "
        search_blob += f"CAT: {category} / {subcategory} || "
        search_blob += f"KEYWORDS: {', '.join(keywords)} || "
        search_blob += f"Q: {question} || A: {answer}"
        
        item = {
            "kb_id": kb_id,
            "domain": domain,
            "scope": scope,
            "category": category,
            "subcategory": subcategory,
            "canonical_question": question,
            "canonical_answer": answer,
            
            "versioning": {
                "valid_from": version or "2025.1",
                "valid_until": None,
                "deprecated": False,
                "superseded_by": None,
                "version_notes": version_info.get("mentions_versions", []),
            },
            
            "rich_metadata": {
                "keywords": keywords,
                "question_type": q_type,
                "source_type": source_type,
                "source_id": str(frame_id) if frame_id else None,
                "scope_confidence": scope_confidence,
                "auto_classified": True,
            },
            
            "search_blob": search_blob,
            "last_updated": today,
            "created_date": today,
        }
        
        canonical.append(item)
    
    # Combine with existing
    if append:
        canonical = existing + canonical
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(canonical, f, indent=2, ensure_ascii=False)
    
    # Summary
    print(f"\n[SUCCESS] Transformed {len(entries)} entries")
    print(f"[OUTPUT] {output_path}")

    # Stats
    scopes = {"platform": 0, "product_specific": 0}
    categories = {}

    for item in canonical[-len(entries):]:  # Only count new entries
        scopes[item["scope"]] = scopes.get(item["scope"], 0) + 1
        cat = item["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n[STATS] Scope Distribution:")
    for scope, count in scopes.items():
        print(f"   {scope}: {count}")

    print(f"\n[STATS] Categories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:10]:
        print(f"   {cat}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Transform knowledge.jsonl to canonical KB format"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input .jsonl file path"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output .json file path (default: auto-generated based on domain)"
    )
    parser.add_argument(
        "-d", "--domain",
        type=str,
        required=True,
        choices=["wms", "planning", "catman", "logistics", "platform"],
        help="Domain tag for entries"
    )
    parser.add_argument(
        "-s", "--source-type",
        type=str,
        default="video_workshop",
        choices=["video_workshop", "document", "meeting", "training", "manual"],
        help="Source type (default: video_workshop)"
    )
    parser.add_argument(
        "-v", "--version",
        type=str,
        default="2025.1",
        help="Product version this applies to (default: 2025.1)"
    )
    parser.add_argument(
        "-a", "--append",
        action="store_true",
        help="Append to existing canonical file instead of overwrite"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / f"data_kb/canonical/RFP_Database_{args.domain.upper()}_CANONICAL.json"
    
    transform_knowledge(
        input_path=input_path,
        output_path=output_path,
        domain=args.domain,
        source_type=args.source_type,
        version=args.version,
        append=args.append,
    )


if __name__ == "__main__":
    main()
