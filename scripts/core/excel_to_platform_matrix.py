"""
excel_to_platform_matrix_v2.py
Convert Platform_Usage_by_Product.xlsx to JSON format with SUB-PRODUCT granularity.

Each sub-product gets its own solution_code for use with --solution flag.

Usage:
    python excel_to_platform_matrix_v2.py --input Platform_Usage_by_Product.xlsx
"""

import pandas as pd
import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any


# Default paths
DEFAULT_INPUT = Path("data_kb/raw/Platform_Usage_by_Product.xlsx")
DEFAULT_OUTPUT = Path("config/platform_matrix.json")
SHEET_NAME = "Usage by Product"


# Excel column definitions
# Format: (column_index, solution_code, display_name, family_code, family_name, cloud_native)
SUBPRODUCT_COLUMNS = [
    # COGNITIVE PLANNING
    (2, "planning", "Demand and Supply Planning", "planning", "Cognitive Planning", True),
    (3, "planning_ibp", "Integrated Business Planning", "planning", "Cognitive Planning", True),
    (4, "planning_pps", "Production Planning and Scheduling", "planning", "Cognitive Planning", True),
    
    # CATEGORY MANAGEMENT
    (5, "catman", "Category Management Suite", "catman", "Category Management Suite", False),
    (6, "catman_assortment", "Strategic Assortment", "catman", "Category Management Suite", False),
    (7, "catman_space", "Strategic Space", "catman", "Category Management Suite", False),
    
    # RETAIL SOLUTIONS
    (8, "retail_ar", "Allocation and Replenishment", "retail", "Retail Solutions", False),
    (9, "retail_ap", "Assortment Planning", "retail", "Retail Solutions", False),
    (10, "retail_clearance", "Clearance Price", "retail", "Retail Solutions", False),
    (11, "retail_demand_edge", "Demand Edge for Retail", "retail", "Retail Solutions", False),
    (12, "retail_markdown", "Fresh Markdown Optimization", "retail", "Retail Solutions", False),
    (13, "retail_ia", "Intelligent Allocation", "retail", "Retail Solutions", False),
    (14, "retail_mfp", "Merchandise Financial Planning", "retail", "Retail Solutions", False),
    
    # SUPPLY CHAIN PLANNING
    (15, "scp", "Demand, D360, Fulfilment, ESP, IO, OO, OP", "scp", "Supply Chain Planning", True),
    (16, "scp_promo_mgmt", "Promotions Management", "scp", "Supply Chain Planning", True),
    (17, "scp_promo_opt", "Promotions Optimization", "scp", "Supply Chain Planning", True),
    (18, "scp_sequencing", "Sequencing", "scp", "Supply Chain Planning", True),
    
    # LOGISTICS
    (19, "logistics", "TM, TP, TMU, Archive", "logistics", "Logistics (TMS)", True),
    (20, "logistics_ba", "BA for TMS", "logistics", "Logistics (TMS)", True),
    (21, "logistics_modeling", "Modeling", "logistics", "Logistics (TMS)", True),
    (22, "logistics_procurement", "Logistics Procurement", "logistics", "Logistics (TMS)", True),
    (23, "logistics_fom", "Freight Order Management", "logistics", "Logistics (TMS)", True),
    (24, "logistics_load", "Load Building", "logistics", "Logistics (TMS)", True),
    (25, "logistics_to_tr", "TO, TR", "logistics", "Logistics (TMS)", True),
    (26, "logistics_carrier", "Carrier Collab, DPD, FT", "logistics", "Logistics (TMS)", True),
    
    # WAREHOUSE
    (27, "wms", "Warehouse Management", "wms", "Warehouse Management", True),
    (28, "wms_native", "Platform Native Warehouse Management", "wms", "Warehouse Management", True),
    (29, "wms_labor", "Warehouse Labor Management", "wms", "Warehouse Management", True),
    (30, "wms_tasking", "Warehouse Tasking", "wms", "Warehouse Management", True),
    (31, "wms_billing", "Billing Management", "wms", "Warehouse Management", True),
    (32, "wms_robotics", "Robotics Hub", "wms", "Warehouse Management", True),
    
    # WORKFORCE
    (33, "workforce", "Workforce Management", "workforce", "Workforce Management", True),
    (34, "workforce_alf", "Advanced Labor Forecasting", "workforce", "Workforce Management", True),
    
    # COMMERCE
    (35, "commerce", "Inventory & Commits Service", "commerce", "Commerce", True),
    (36, "commerce_orders", "Order Services", "commerce", "Commerce", True),
    
    # DODDLE
    (37, "doddle", "Returns Management", "doddle", "Doddle", True),
    
    # NETWORK
    (38, "network", "Command Center", "network", "Network", True),
    
    # FLEXIS
    (39, "flexis", "Order Sequencing", "flexis", "Flexis", False),
    (40, "flexis_slotting", "Order Slotting", "flexis", "Flexis", False),
    (41, "flexis_other", "Other Products", "flexis", "Flexis", False),
    
    # CONTROL TOWER
    (42, "control_tower", "Control Tower Visibility", "control_tower", "Control Tower", True),
]


def parse_cell_status(value) -> Dict[str, Any]:
    """
    Parse cell value to status dict.
    
    Returns:
        {
            "status": "native" | "coming" | "infrastructure",
            "available": True/False (for native),
            "note": optional note for coming
        }
    """
    if pd.isna(value):
        return {
            "status": "infrastructure",
            "available": False,
            "note": None
        }
    
    val_str = str(value).strip()
    
    # Check for checkmark (letter 'a' in Wingdings or actual checkmark)
    if val_str == "a" or val_str == "✓" or val_str == "✔":
        return {
            "status": "native",
            "available": True,
            "note": None
        }
    
    # Check for "Coming" pattern
    if "coming" in val_str.lower():
        # Extract version if present but don't expose it
        return {
            "status": "coming",
            "available": False,
            "note": "planned"  # Don't expose version details
        }
    
    # Default to infrastructure
    return {
        "status": "infrastructure", 
        "available": False,
        "note": None
    }


def convert_excel_to_json(input_path: Path, output_path: Path):
    """Convert Excel matrix to JSON with sub-product granularity."""
    
    print(f"[INFO] Reading {input_path}...")
    df = pd.read_excel(input_path, sheet_name=SHEET_NAME, header=None)
    
    # Build solution lookup
    solutions = {}
    
    for col, solution_code, display_name, family_code, family_name, cloud_native in SUBPRODUCT_COLUMNS:
        solutions[solution_code] = {
            "solution_code": solution_code,
            "display_name": display_name.replace('\n', ' ').strip(),
            "family_code": family_code,
            "family_name": family_name,
            "cloud_native": cloud_native,
            "excel_column": col,
            "services": {}
        }
    
    # Extract services (rows 5-24, which is index 4-23)
    services_list = []
    
    for row in range(5, 25):  # Excel rows 6-25 (1-indexed), so index 5-24
        raw_name = df.iloc[row, 1]
        if pd.isna(raw_name):
            continue
        
        parts = str(raw_name).split('\n')
        service_name = parts[0].strip()
        service_desc = ' '.join(parts[1:]).strip() if len(parts) > 1 else ""
        
        service = {
            "name": service_name,
            "description": service_desc,
            "solutions": {}
        }
        
        # Get status for each solution (sub-product)
        for col, solution_code, _, _, _, _ in SUBPRODUCT_COLUMNS:
            cell_value = df.iloc[row, col]
            status = parse_cell_status(cell_value)
            service["solutions"][solution_code] = status
            
            # Also add to solution's services dict
            solutions[solution_code]["services"][service_name] = status
        
        services_list.append(service)
    
    # Build family summary
    families = {}
    for col, solution_code, display_name, family_code, family_name, cloud_native in SUBPRODUCT_COLUMNS:
        if family_code not in families:
            families[family_code] = {
                "name": family_name,
                "cloud_native": cloud_native,
                "solutions": []
            }
        families[family_code]["solutions"].append(solution_code)
    
    # Build output
    output = {
        "metadata": {
            "source": str(input_path.name),
            "generated": datetime.now().isoformat(),
            "version": "2.0",
            "description": "Platform services matrix with sub-product granularity"
        },
        "product_families": families,
        "solutions": solutions,
        "platform_services": services_list
    }
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS] Converted {len(services_list)} services x {len(solutions)} solutions")
    print(f"[INFO] Output: {output_path}")
    
    # Print solution codes for reference
    print(f"\n[INFO] Available --solution flags:")
    current_family = None
    for col, solution_code, display_name, family_code, family_name, _ in SUBPRODUCT_COLUMNS:
        if family_code != current_family:
            print(f"\n  {family_name}:")
            current_family = family_code
        print(f"    --solution {solution_code:25s} # {display_name[:40]}")
    
    # Summary stats
    print(f"\n[INFO] Summary:")
    print(f"  Total solutions (sub-products): {len(solutions)}")
    print(f"  Total services: {len(services_list)}")
    print(f"  Product families: {len(families)}")


def main():
    parser = argparse.ArgumentParser(description="Convert Platform Usage Excel to JSON (v2 - sub-product granularity)")
    parser.add_argument("-i", "--input", type=str, default=str(DEFAULT_INPUT),
                       help=f"Input Excel file (default: {DEFAULT_INPUT})")
    parser.add_argument("-o", "--output", type=str, default=str(DEFAULT_OUTPUT),
                       help=f"Output JSON file (default: {DEFAULT_OUTPUT})")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1
    
    convert_excel_to_json(input_path, output_path)
    return 0


if __name__ == "__main__":
    exit(main())