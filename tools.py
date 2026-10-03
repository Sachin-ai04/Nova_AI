"""
tools.py — LangChain tools for the Nova AI Customer Support agent.

Data layer
----------
All data lives in flat CSV files under data/public/.  Every tool loads
the relevant CSV(s) at call-time via pandas so that hot-reloads during
development always reflect the latest data on disk.

Tool index
----------
1. get_order_by_id        – full order record + all line-items for one order
2. get_customer_by_id     – full customer profile for one customer
3. calculate_refund_cap   – policy-aware refund ceiling for one order/item
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Path constants — all relative to this file's directory
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent / "data" / "public"
_ORDERS_CSV       = _DATA_DIR / "orders.csv"
_ORDER_ITEMS_CSV  = _DATA_DIR / "order_items.csv"
_CUSTOMERS_CSV    = _DATA_DIR / "customers.csv"
_PRODUCTS_CSV     = _DATA_DIR / "products.csv"


# ---------------------------------------------------------------------------
# Internal helpers (not exposed as tools)
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV into a DataFrame, raising a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}.  "
            "Ensure the data/public/ directory is present next to tools.py."
        )
    return pd.read_csv(path, dtype=str)


def _row_to_dict(row: pd.Series) -> dict:
    """Convert a DataFrame row to a plain dict, replacing NaN with None."""
    return {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
            for k, v in row.items()}


# ---------------------------------------------------------------------------
# Tool 1 — get_order_by_id
# ---------------------------------------------------------------------------

@tool
def get_order_by_id(order_id: str) -> dict:
    """
    Retrieve the full order record and all associated line-items for a single
    order, identified by its NovaMart order ID (e.g. "ORD-000042").

    ── WHEN TO CALL THIS TOOL ─────────────────────────────────────────────────

    Call this tool whenever the customer mentions a specific order ID, asks
    about the status of an order, wants to know about delivery, payment,
    tracking, cancellation, refund, or return status for an order, or when
    you need any order field to apply a policy.

    Do NOT call this tool speculatively for every message.  Only call it once
    you have an explicit order ID from the customer or from a prior tool call.
    If the customer hasn't provided an order ID yet, ask for it first.

    ── INPUT ──────────────────────────────────────────────────────────────────

    order_id : str
        The NovaMart order identifier.  Always in the format "ORD-NNNNNN"
        (e.g. "ORD-000001", "ORD-012345").  Strip any surrounding whitespace
        before passing.  The lookup is case-sensitive; always pass uppercase.

    ── OUTPUT ─────────────────────────────────────────────────────────────────

    Returns a dict with two top-level keys:

    "order" : dict | None
        The single matching order row, or None if no order with that ID exists.
        Fields returned (all strings unless noted):

        order_id              – e.g. "ORD-000001"
        customer_id           – e.g. "CUST-00615" (use get_customer_by_id for details)
        order_date            – ISO datetime string, IST, e.g. "2026-01-01 16:35:39"
                                IMPORTANT: determines which refund/return policy version
                                applies (v1 for dates before 2026-06-01; v2 from 2026-06-01).
        order_status          – one of: "pending" | "processing" | "shipped" |
                                "delivered" | "cancelled" | "return_initiated" |
                                "returned" | "replaced"
        payment_method        – one of: "cash_on_delivery" | "credit_card" |
                                "debit_card" | "UPI" | "net_banking" | "wallet"
                                NOTE: cash_on_delivery refunds go to NovaMart wallet
                                or verified bank account, not back to card.
        payment_status        – one of: "pending" | "paid" | "failed" | "refunded"
        subtotal              – item total before discount/fees (INR, as string)
        discount              – order-level discount applied (INR)
        shipping_fee          – shipping charged (INR, 0 means free shipping)
        tax                   – GST charged on the order (INR)
        total_amount          – final amount paid by the customer (INR)
                                CRITICAL for refund approval threshold checks:
                                v1 threshold = INR 1,00,000 (needs human above);
                                v2 threshold = INR 75,000  (needs human above).
        shipping_address      – full street address string
        city, state           – delivery location
        estimated_delivery_date – "YYYY-MM-DD" or None
        actual_delivery_date  – "YYYY-MM-DD" or None; None means not yet delivered.
                                Day 0 for refund/return window calculations.
        tracking_number       – courier tracking ID or None
        courier               – courier name or None
        delivery_status       – one of: "pending" | "in_transit" | "out_for_delivery" |
                                "delivered" | "failed_attempt" | "returned_to_origin"
        delivery_otp_verified – "true" | "false"
                                If "true" and customer disputes delivery, escalate to human.
        cancellation_status   – one of: "none" | "requested" | "approved" | "rejected"
        refund_status         – one of: "none" | "pending" | "processing" |
                                "processed" | "failed"

    "items" : list[dict]
        Zero or more line-item rows for this order.  Each item dict contains:

        order_item_id  – e.g. "OI-000001"
        order_id       – parent order
        product_id     – e.g. "PROD-00017"; cross-reference products.csv for
                         category, returnable flag, warranty, etc.
        quantity       – number of units (string; cast to int as needed)
        unit_price     – listed price per unit before discount (INR)
        discount       – per-item discount (INR)
        final_price    – amount customer actually paid for this item (INR)
                         Use this value as the base for refund amount calculations
                         (add 18% GST on top per policy sections 4.1).
        item_status    – one of: "pending" | "processing" | "shipped" |
                         "delivered" | "cancelled" | "return_initiated" |
                         "returned" | "replaced"
        return_status  – one of: "none" | "requested" | "approved" | "rejected" |
                         "received" | "qc_passed" | "qc_failed"
        refund_amount  – refund already issued for this item (INR); may be "0"

    ── RETURN VALUE WHEN NOT FOUND ────────────────────────────────────────────

    If no order matches the given ID:
        {"order": None, "items": [], "error": "Order ORD-XXXXXX not found."}

    ── USAGE NOTES FOR THE LLM ────────────────────────────────────────────────

    1. Policy version selection
       Compare order["order_date"][:10] with "2026-06-01".
       If order_date < "2026-06-01"  → apply Refund Policy v1 (return window 10/15 days,
                                        approval threshold INR 1,00,000, no restocking fee).
       If order_date >= "2026-06-01" → apply Refund Policy v2 (return window 7/10 days,
                                        approval threshold INR 75,000, restocking fee on
                                        Laptops/Tablets/Cameras/Monitors for change-of-mind).

    2. Refund window calculation
       Use order["actual_delivery_date"] as day 0.
       Count calendar days to today's date to determine eligibility.
       Do NOT use estimated_delivery_date for window calculations.

    3. Approval threshold
       Cast order["total_amount"] to float and compare against the policy threshold
       (not the item's final_price). Even a small-item refund from a high-value order
       requires human approval if total_amount exceeds the threshold.

    4. Escalation triggers
       Always escalate to a human if delivery_otp_verified == "true" and the customer
       claims the order was not delivered.

    Example call:
        get_order_by_id("ORD-000042")
    """
    orders_df = _load_csv(_ORDERS_CSV)
    items_df  = _load_csv(_ORDER_ITEMS_CSV)

    order_id = order_id.strip()

    order_rows = orders_df[orders_df["order_id"] == order_id]
    if order_rows.empty:
        return {
            "order": None,
            "items": [],
            "error": f"Order {order_id} not found.",
        }

    order = _row_to_dict(order_rows.iloc[0])
    items = [
        _row_to_dict(row)
        for _, row in items_df[items_df["order_id"] == order_id].iterrows()
    ]

    return {"order": order, "items": items}


# ---------------------------------------------------------------------------
# Tool 2 — get_customer_by_id
# ---------------------------------------------------------------------------

@tool
def get_customer_by_id(customer_id: str) -> dict:
    """
    Retrieve the complete customer profile for a single NovaMart customer,
    identified by their customer ID (e.g. "CUST-00001").

    ── WHEN TO CALL THIS TOOL ─────────────────────────────────────────────────

    Call this tool when:
    • You need to verify account status before processing any refund or return.
      Suspended accounts must be escalated to human review immediately.
    • You need the customer's loyalty_tier to apply loyalty window extensions
      for refund/return eligibility (Gold +2 days, Platinum +3 days on the
      change-of-mind window only).
    • You need demographic or contact details for communication or verification.
    • An order lookup (get_order_by_id) returned a customer_id and you need the
      full profile to continue policy checks.

    Do NOT call this tool for every message.  If the customer's ID is not yet
    known, derive it from the order (order["customer_id"]) instead of asking
    the customer for their internal ID.

    ── INPUT ──────────────────────────────────────────────────────────────────

    customer_id : str
        The NovaMart customer identifier, always in the format "CUST-NNNNN"
        (e.g. "CUST-00001").  Strip whitespace; pass uppercase.

    ── OUTPUT ─────────────────────────────────────────────────────────────────

    Returns a dict with one top-level key:

    "customer" : dict | None
        The matching customer row, or None if not found.
        Fields returned (all strings):

        customer_id       – e.g. "CUST-00001"
        first_name        – customer's first name
        last_name         – customer's last name
        email             – registered e-mail address
        phone             – registered mobile number (with country code)
        gender            – "male" | "female" | "other" | None
        date_of_birth     – "YYYY-MM-DD" or None
        city, state       – registered city and state (may differ from delivery address)
        pincode           – postal code of registered address
        address           – registered street address (not necessarily the delivery address)
        customer_since    – "YYYY-MM-DD" date of account creation
        customer_segment  – one of: "new" | "regular" | "premium" | "high_value"
        account_status    – one of: "active" | "suspended" | "closed"
                            CRITICAL: if account_status == "suspended", do NOT initiate
                            any refund or return.  Immediately escalate to Trust & Safety
                            (human review) and do not disclose the suspension reason.
        preferred_language – e.g. "English", "Hindi", "Hinglish", "Tamil", etc.
                             Respond to the customer in their preferred language when possible.
        total_orders      – lifetime order count (string; cast to int as needed)
        total_spend       – cumulative spend in INR (string; cast to float as needed)
        loyalty_tier      – one of: "bronze" | "silver" | "gold" | "platinum"
                            Loyalty tier affects the change-of-mind refund/return window:
                              bronze / silver → no extension (standard window applies)
                              gold            → +2 calendar days on change-of-mind window
                              platinum        → +3 calendar days on change-of-mind window
                            The extension applies to change-of-mind ONLY, not to defect,
                            damage, wrong item, missing item, or lost order claims.

    ── RETURN VALUE WHEN NOT FOUND ────────────────────────────────────────────

    If no customer matches:
        {"customer": None, "error": "Customer CUST-XXXXX not found."}

    ── USAGE NOTES FOR THE LLM ────────────────────────────────────────────────

    1. Account status gate
       Before taking ANY action on a refund, return, or replacement request,
       check account_status.  Only "active" accounts may proceed without
       escalation.

    2. Loyalty window extension (change-of-mind only)
       Extended window = base_window + loyalty_days
       where loyalty_days = 2 if tier == "gold" else 3 if tier == "platinum" else 0.
       This extension applies to the change-of-mind window ONLY.
       Defect/damage/wrong-item windows are NOT extended by loyalty.

    3. Preferred language
       Use preferred_language to tailor your response language or to note that
       the customer prefers a regional language when escalating to a human agent.

    4. Pattern detection hint
       total_orders and total_spend can signal repeat or high-value customers,
       but do NOT use them alone to flag fraud.  Refer to the Customer Escalation
       Policy for pattern-of-claims guidance.

    Example call:
        get_customer_by_id("CUST-00615")
    """
    customers_df = _load_csv(_CUSTOMERS_CSV)

    customer_id = customer_id.strip()

    rows = customers_df[customers_df["customer_id"] == customer_id]
    if rows.empty:
        return {
            "customer": None,
            "error": f"Customer {customer_id} not found.",
        }

    return {"customer": _row_to_dict(rows.iloc[0])}


# ---------------------------------------------------------------------------
# Tool 3 — calculate_refund_cap
# ---------------------------------------------------------------------------

@tool
def calculate_refund_cap(
    order_id: str,
    order_item_id: str,
    reason: str,
    request_date: Optional[str] = None,
) -> dict:
    """
    Calculate the maximum refund amount (the "cap") for a specific line-item
    on a NovaMart order, applying the correct policy version, restocking fee
    (if applicable), and shipping fee eligibility — and returns whether human
    approval is required before the refund can be released.

    This tool DOES NOT create or approve any refund.  It only computes numbers
    and flags.  A human or a downstream action must act on the result.

    ── WHEN TO CALL THIS TOOL ─────────────────────────────────────────────────

    Call this tool when:
    • The customer has requested a refund and you have confirmed the order ID,
      specific item ID, and the reason for the refund request.
    • You need to tell the customer the exact amount they are eligible to receive.
    • You need to know whether the refund requires human approval before proceeding.

    Do NOT call this tool before calling get_order_by_id and get_customer_by_id —
    you need the order total, item final_price, order date, customer loyalty_tier,
    and account_status from those tools first.

    Do NOT call this tool if the refund eligibility window has already expired, or
    if the item is not returnable for a change-of-mind reason.  Check eligibility
    first; only call this tool when refund eligibility is confirmed.

    ── INPUTS ─────────────────────────────────────────────────────────────────

    order_id : str
        The NovaMart order ID, e.g. "ORD-000042".

    order_item_id : str
        The specific line-item ID within the order, e.g. "OI-000007".
        Must belong to the order specified in order_id.
        If the customer is returning all items, call this tool once per item.

    reason : str
        The reason for the refund.  Must be one of the following canonical values
        (case-insensitive, the tool normalises internally):

        "change_of_mind"        – customer changed their mind; item must be
                                  returnable, unused, and in original packaging.
        "defective"             – product is defective or dead on arrival.
        "damaged_in_transit"    – product was damaged during shipping.
        "wrong_item"            – wrong product was delivered.
        "missing_item"          – one or more items were missing from the package.
        "not_delivered"         – order was not delivered / lost in transit.
        "duplicate_charge"      – customer was charged twice or overcharged.
        "cancelled_prepaid"     – order cancelled before shipment (prepaid orders).

        Any other value will be treated as unknown and the tool will return an
        error asking you to use a canonical reason.

    request_date : str | None
        The date the customer originally raised the refund/return/damage request,
        in "YYYY-MM-DD" format.  Use today's date if not specified.

        IMPORTANT: if a support ticket exists (from a prior tool call or context)
        with a created_at date inside the eligibility window, pass THAT date as
        request_date — even if today is outside the window.  Per both policy
        versions section 7.2, eligibility is judged on the original request date
        when the delay in completing the return was caused by NovaMart or the courier.

    ── OUTPUT ─────────────────────────────────────────────────────────────────

    Returns a dict with the following fields:

    "order_id"           : str  – echoed from input
    "order_item_id"      : str  – echoed from input
    "reason"             : str  – normalised reason string
    "policy_version"     : str  – "v1" or "v2" (determined by order_date)

    "item_final_price"   : float – order_items.final_price for this item (INR)
    "gst_on_item"        : float – 18% GST on final_price, included in refund (INR)
    "item_refund_base"   : float – final_price + gst_on_item (INR)

    "restocking_fee"     : float – deducted from refund for change-of-mind returns
                                   under v2 for Laptops, Tablets, Cameras, Monitors.
                                   = 5% of item_refund_base, capped at INR 2,500.
                                   0.0 for all other cases.
    "restocking_fee_note": str   – human-readable explanation of the fee (or empty).

    "shipping_refund"    : float – shipping fee refunded (INR).
                                   Non-zero only when:
                                     (a) cancelled_prepaid + full order, OR
                                     (b) reason is not_delivered, OR
                                     (c) wrong_item / defective / damaged_in_transit
                                         AND this is the ONLY item in the order (full return).
                                   For partial returns, always 0.
                                   NOTE: this tool assumes a partial return unless
                                   is_full_order_return is set.  If the caller knows
                                   this is a full-order return, re-call with the flag
                                   (see is_full_order_return note below).

    "refund_cap"         : float – maximum total refund for this item (INR).
                                   = item_refund_base - restocking_fee + shipping_refund.
                                   Capped at orders.total_amount across all items.

    "already_refunded"   : float – refund_amount already processed for this item (INR).
    "net_refundable"     : float – refund_cap - already_refunded; the actual amount
                                   that can still be issued.  Never negative.

    "needs_human_approval" : bool
        True when:
          • order total_amount > INR 75,000 (v2) or > INR 1,00,000 (v1), OR
          • customer account_status == "suspended" (call get_customer_by_id first), OR
          • any other escalation condition from policy section 9.
        When True, do NOT promise the customer that the refund is approved.
        Inform them that it has been referred to a refund approver.

    "approval_threshold" : float – policy threshold used (75000 or 100000)
    "order_total"        : float – orders.total_amount used for threshold check

    "eligible"           : bool  – True if the item appears eligible for a refund
                                   based on item status and refund_status.
                                   False if refund_status is already "processed".

    "notes"              : list[str] – list of important caveats or policy reminders
                                       the agent should convey to the customer.

    "error"              : str | None – set if the order, item, or reason is invalid;
                                        None otherwise.

    ── POLICY RULES ENCODED IN THIS TOOL ─────────────────────────────────────

    Policy version (determined by order_date):
      v1 → orders placed 2026-01-01 to 2026-05-31 (inclusive)
  	  v2 → orders placed 2026-06-01 onwards

    Refund amount formula (both versions):
      item_refund_base = final_price + (final_price × 0.18)   [item + GST]
      restocking_fee   = 0 (v1) or 5% of item_refund_base capped at 2500 (v2,
                         change_of_mind only, for Laptops/Tablets/Cameras/Monitors)
      refund_cap       = item_refund_base - restocking_fee + shipping_refund
      net_refundable   = max(0, refund_cap - already_refunded)

    Approval thresholds:
      v1: order total > INR 1,00,000 → needs human
      v2: order total > INR 75,000   → needs human
      Threshold is on order total, NOT item value.

    Restocking fee categories (v2, change_of_mind only):
      Fee applies to: Laptops, Tablets, Cameras, Monitors
      Fee = min(item_refund_base × 0.05, 2500)
      Fee does NOT apply to: defective, damaged, wrong item, or any v1 order.

    ── USAGE NOTES FOR THE LLM ────────────────────────────────────────────────

    1. Always confirm eligibility BEFORE calling this tool.
       Check the refund window (using actual_delivery_date and request_date)
       and the item's returnable flag (for change_of_mind) before invoking.

    2. Communicate net_refundable to the customer, not refund_cap.
       If already_refunded > 0, explain that a partial refund was already issued.

    3. If needs_human_approval is True, say:
       "Your refund request has been forwarded to our refund approval team.
        You will receive a confirmation within 1-2 business days."
       Do NOT say "your refund is approved."

    4. For partial returns (one item from a multi-item order), shipping is never
       refunded.  Do not add shipping manually — this tool handles it.

    5. Always show the restocking_fee_note to the customer if restocking_fee > 0.

    Example call:
        calculate_refund_cap(
            order_id="ORD-000042",
            order_item_id="OI-000099",
            reason="change_of_mind",
            request_date="2026-08-15",
        )
    """
    # ── Load data ──────────────────────────────────────────────────────────
    orders_df   = _load_csv(_ORDERS_CSV)
    items_df    = _load_csv(_ORDER_ITEMS_CSV)
    products_df = _load_csv(_PRODUCTS_CSV)

    order_id      = order_id.strip()
    order_item_id = order_item_id.strip()
    reason        = reason.strip().lower()

    VALID_REASONS = {
        "change_of_mind", "defective", "damaged_in_transit",
        "wrong_item", "missing_item", "not_delivered",
        "duplicate_charge", "cancelled_prepaid",
    }
    if reason not in VALID_REASONS:
        return {
            "error": (
                f"Unknown reason '{reason}'. "
                f"Use one of: {', '.join(sorted(VALID_REASONS))}."
            )
        }

    # ── Fetch order ────────────────────────────────────────────────────────
    order_rows = orders_df[orders_df["order_id"] == order_id]
    if order_rows.empty:
        return {"error": f"Order {order_id} not found.", "order_id": order_id}
    order = order_rows.iloc[0]

    # ── Fetch target item ──────────────────────────────────────────────────
    all_items = items_df[items_df["order_id"] == order_id]
    item_rows = all_items[all_items["order_item_id"] == order_item_id]
    if item_rows.empty:
        return {
            "error": f"Item {order_item_id} not found in order {order_id}.",
            "order_id": order_id,
            "order_item_id": order_item_id,
        }
    item = item_rows.iloc[0]

    # ── Already fully refunded? ────────────────────────────────────────────
    already_refunded = float(item.get("refund_amount") or 0)
    eligible = not (
        str(order.get("refund_status", "")).lower() == "processed"
        and already_refunded > 0
    )

    # ── Policy version ─────────────────────────────────────────────────────
    order_date_str = str(order["order_date"])[:10]  # "YYYY-MM-DD"
    policy_version = "v1" if order_date_str < "2026-06-01" else "v2"
    approval_threshold = 100_000.0 if policy_version == "v1" else 75_000.0

    # ── Core refund maths ──────────────────────────────────────────────────
    final_price      = float(item.get("final_price") or 0)
    gst_on_item      = round(final_price * 0.18, 2)
    item_refund_base = round(final_price + gst_on_item, 2)

    # ── Restocking fee (v2, change_of_mind, specific categories only) ──────
    restocking_fee      = 0.0
    restocking_fee_note = ""
    if policy_version == "v2" and reason == "change_of_mind":
        product_id = str(item.get("product_id", ""))
        prod_rows  = products_df[products_df["product_id"] == product_id]
        if not prod_rows.empty:
            category = str(prod_rows.iloc[0].get("category", "")).strip()
            FEE_CATEGORIES = {"Laptops", "Tablets", "Cameras", "Monitors"}
            if category in FEE_CATEGORIES:
                fee_raw        = item_refund_base * 0.05
                restocking_fee = round(min(fee_raw, 2500.0), 2)
                restocking_fee_note = (
                    f"A restocking fee of INR {restocking_fee:,.2f} applies "
                    f"to change-of-mind returns on {category} under Policy v2 "
                    f"(5% of item refund, capped at INR 2,500)."
                )

    # ── Shipping refund ────────────────────────────────────────────────────
    order_shipping_fee  = float(order.get("shipping_fee") or 0)
    is_single_item_order = len(all_items) == 1
    shipping_refund = 0.0
    if reason in ("cancelled_prepaid", "not_delivered"):
        shipping_refund = order_shipping_fee
    elif reason in ("wrong_item", "defective", "damaged_in_transit") and is_single_item_order:
        shipping_refund = order_shipping_fee
    # Partial returns never get shipping refund (left as 0.0)

    # ── Refund cap & net refundable ────────────────────────────────────────
    order_total      = float(order.get("total_amount") or 0)
    refund_cap_raw   = item_refund_base - restocking_fee + shipping_refund
    refund_cap       = round(min(refund_cap_raw, order_total), 2)
    net_refundable   = round(max(0.0, refund_cap - already_refunded), 2)

    # ── Approval check ─────────────────────────────────────────────────────
    needs_human_approval = order_total > approval_threshold

    # ── Notes ──────────────────────────────────────────────────────────────
    notes: list[str] = []
    if not eligible:
        notes.append(
            "A refund has already been fully processed for this item.  "
            "No further refund can be issued."
        )
    if needs_human_approval:
        notes.append(
            f"Order total (INR {order_total:,.0f}) exceeds the "
            f"INR {approval_threshold:,.0f} threshold under Policy {policy_version}.  "
            "This refund MUST be approved by a human Refund Approver before release."
        )
    if reason == "change_of_mind" and policy_version == "v2":
        notes.append(
            "Under Policy v2, change-of-mind returns are subject to the 7-day window "
            "(standard) with Gold +2 / Platinum +3 day loyalty extension.  "
            "Verify eligibility window before confirming."
        )
    if reason == "change_of_mind" and policy_version == "v1":
        notes.append(
            "Under Policy v1, the change-of-mind return window is 10 days "
            "(Gold +2 / Platinum +3).  No restocking fee applies."
        )
    if net_refundable == 0.0 and already_refunded > 0:
        notes.append(
            f"INR {already_refunded:,.2f} has already been refunded for this item, "
            "which equals the refund cap.  No additional amount is payable."
        )
    if shipping_refund == 0.0 and is_single_item_order and reason not in (
        "cancelled_prepaid", "not_delivered", "wrong_item", "defective", "damaged_in_transit"
    ):
        notes.append(
            "Shipping fee is not refundable for this reason under NovaMart policy."
        )

    return {
        "order_id":             order_id,
        "order_item_id":        order_item_id,
        "reason":               reason,
        "policy_version":       policy_version,
        "item_final_price":     final_price,
        "gst_on_item":          gst_on_item,
        "item_refund_base":     item_refund_base,
        "restocking_fee":       restocking_fee,
        "restocking_fee_note":  restocking_fee_note,
        "shipping_refund":      shipping_refund,
        "refund_cap":           refund_cap,
        "already_refunded":     already_refunded,
        "net_refundable":       net_refundable,
        "needs_human_approval": needs_human_approval,
        "approval_threshold":   approval_threshold,
        "order_total":          order_total,
        "eligible":             eligible,
        "notes":                notes,
        "error":                None,
    }
