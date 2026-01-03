# Blue Yonder Platform - Response Context Guide

> This document provides context for generating RFP responses. 
> It explains the platform architecture and how to frame answers based on integration levels.

---

## Platform Architecture Overview

Blue Yonder Platform is the **infrastructure layer** that provides shared services to all products.

```
┌─────────────────────────────────────────────────────────────────┐
│                    BLUE YONDER PLATFORM                         │
│                    (Infrastructure Layer)                       │
│                                                                 │
│  Platform Services:                                             │
│  • Authentication (SSO)    • API Management                     │
│  • Authorization (RBAC)    • Workflow Orchestrator              │
│  • Data Cloud              • Runtime Environment                │
│  • Bulk Ingestion          • ML Studio                          │
│  • Streaming               • And more...                        │
│                                                                 │
│  → These services are available to ALL platform customers       │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │  Planning   │     │     WMS     │     │   Retail    │
   │  (native)   │     │  (native)   │     │  (enabled)  │
   └─────────────┘     └─────────────┘     └─────────────┘
```

---

## Product Families

| Family | Cloud Native | Products |
|--------|--------------|----------|
| **Cognitive Planning** | Yes | Demand and Supply Planning, IBP, Production Planning |
| **Warehouse Management** | Yes | WMS, Platform Native WMS, Labor Management, Tasking, Billing, Robotics Hub |
| **Logistics (TMS)** | Yes | TM, TP, TMU, BA for TMS, Modeling, Freight Order Management |
| **Category Management** | No | Category Management Suite, Strategic Assortment, Strategic Space |
| **Retail Solutions** | No | Allocation & Replenishment, Assortment Planning, Clearance, Markdown |
| **Supply Chain Planning** | Yes | Demand, D360, Fulfilment, ESP, Promotions |
| **Workforce Management** | Yes | Workforce Management, Advanced Labor Forecasting |
| **Commerce** | Yes | Inventory & Commits Service, Order Services |
| **Control Tower** | Yes | Control Tower Visibility |
| **Flexis** | No | Order Sequencing, Order Slotting |

---

## Integration Levels

Each platform service can have different integration levels with each product:

| Level | Internal Status | Meaning |
|-------|-----------------|---------|
| **Native** | ✓ (checkmark) | Service is configured and integrated with product processes out-of-the-box |
| **Planned** | Coming | Service available at platform level; native integration is on the roadmap |
| **Infrastructure** | Blank/Empty | Service available at platform level; customers can leverage it |

---

## Response Strategy

### CRITICAL RULES:

1. **NEVER say** "not available", "doesn't support", "cannot", "is not possible"
2. **ALWAYS respond positively** - the capability EXISTS on the platform
3. **Adjust framing** based on integration level (see below)
4. **NEVER mention** specific versions, release dates, or roadmap timelines
5. **Focus on platform** when integration is not native

---

## Response Templates by Integration Level

### When status is ✓ (Native Integration)

**Template:**
> Use the answer from Knowledge Base WITHOUT any modification. Do NOT mention "platform" or "configured through". Answer naturally as if the capability is a native part of the product.

**Examples:**
- "Blue Yonder's planning solutions feature a web-based graphical user interface (GUI) designed to be engaging and approachable..."
- "The solution supports multiple users working in parallel, with robust collaboration capabilities..."
- "Users can customize and configure the user interface to their needs..."

**Key:** Answer directly from KB. NO platform language. The feature IS native to the product.

---

### When status is "Coming" (Planned Integration)

**Template:**
> "Blue Yonder Platform supports this functionality on an infrastructure level and full native integration is planned."

**Examples:**
- "Blue Yonder Platform supports workflow orchestration on an infrastructure level and full native integration is planned."
- "Blue Yonder Platform supports this functionality on an infrastructure level and full native integration is planned."

**Key:** 
- Do NOT mention the specific product
- Do NOT mention versions or dates
- DO mention that native integration is planned

---

### When status is Blank/Empty (Infrastructure Level)

**Template:**
> "Blue Yonder Platform supports this functionality on an infrastructure level."

**Examples:**
- "Blue Yonder Platform supports this functionality on an infrastructure level."
- "Blue Yonder Platform supports workflow automation on an infrastructure level."

**Key:**
- Do NOT mention the specific product
- Do NOT mention any plans
- Simply state platform supports it

---

## Response Examples

### Example 1: Workflow Automation for WMS

**Question:** "Does Warehouse Management support workflow automation?"

| If Status | Response |
|-----------|----------|
| ✓ | "Blue Yonder's Warehouse Management solution provides comprehensive workflow automation capabilities, enabling process orchestration, task automation, and exception handling across warehouse operations. Users can define automated workflows to streamline picking, packing, and shipping processes." |
| Coming | "Blue Yonder Platform supports workflow automation on an infrastructure level and full native integration is planned. Customers can leverage the platform's workflow orchestration capabilities to automate business processes." |
| Blank | "Blue Yonder Platform supports workflow automation on an infrastructure level. Customers can leverage the platform's workflow orchestration capabilities to automate business processes." |

---

### Example 2: ML Capabilities for Planning

**Question:** "Does the planning solution have machine learning capabilities?"

| If Status | Response |
|-----------|----------|
| ✓ | "Blue Yonder's Cognitive Planning solution leverages advanced machine learning for demand sensing, forecast accuracy improvement, and anomaly detection. The ML models continuously learn from historical data to improve prediction accuracy over time." |
| Coming | "Blue Yonder Platform supports machine learning capabilities on an infrastructure level and full native integration is planned." |
| Blank | "Blue Yonder Platform supports machine learning capabilities on an infrastructure level." |

---

### Example 3: User Interface

**Question:** "Does the system support web-based user interface?"

| If Status | Response |
|-----------|----------|
| ✓ | "Blue Yonder's planning solutions feature a web-based graphical user interface (GUI) designed to be engaging, approachable, and compliant with W3C Level 2 Accessibility standards. The interface supports intuitive navigation with drag-and-drop functionality, spreadsheets for data manipulation, and customizable views." |
| Coming | "Blue Yonder Platform supports web-based user interface on an infrastructure level and full native integration is planned." |
| Blank | "Blue Yonder Platform supports web-based user interface on an infrastructure level." |

---

## Additional Guidelines

### Elaborating on Answers

After the core response (based on integration level), you MAY elaborate with:
- General platform capabilities
- Technical details from the knowledge base
- Benefits and use cases

**But NEVER:**
- Claim native integration when status is Coming or Blank
- Mention specific products when status is Coming or Blank
- Give timelines or version numbers
- Say anything is "not available" or "not supported"

### Handling Follow-up Questions

If customer asks for more details about a non-native integration:
- Continue focusing on platform capabilities
- Explain what the platform offers
- Avoid making promises about product-specific features

### When Unsure

If integration status is unclear:
- Default to the "Infrastructure Level" response
- Focus on platform capabilities
- Keep response positive and helpful

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│ INTEGRATION STATUS → RESPONSE PATTERN                          │
├─────────────────────────────────────────────────────────────────┤
│ ✓ Native    → USE KB ANSWER AS-IS. No platform language.       │
│               Answer naturally, the feature IS native.         │
├─────────────────────────────────────────────────────────────────┤
│ Coming      → "Blue Yonder Platform supports this              │
│               functionality on an infrastructure level         │
│               and full native integration is planned."         │
├─────────────────────────────────────────────────────────────────┤
│ Blank       → "Blue Yonder Platform supports this              │
│               functionality on an infrastructure level."       │
└─────────────────────────────────────────────────────────────────┘

IMPORTANT: 
- Native (✓) = DO NOT modify KB answer, DO NOT add "platform" language
- Only Coming/Blank = add platform infrastructure messaging
```

---

## Document Version

- Version: 1.0
- Last Updated: 2024-12-31
- Purpose: LLM context injection for solution-specific RFP responses
