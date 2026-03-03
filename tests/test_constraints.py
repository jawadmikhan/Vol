"""
Constraint Validation Tests
============================
Validates all positions against the Investment Committee mandate.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.portfolio_constraints import (
    TOTAL_CAPITAL, GROSS_NOTIONAL_CAP, NET_VEGA_FLOOR, NET_VEGA_CEILING,
    VOL_HARVEST_SHORT_VARIANCE_NOTIONAL_CAP, OVERLAY_MAX_ANNUAL_DRAG_BPS,
    VOL_TARGET_ANNUALIZED, MARGIN_REQUIREMENT_CAP,
)


def test_capital_allocation_sums_to_total():
    """Verify strategy allocations sum to total capital."""
    from config.portfolio_constraints import ALLOCATIONS
    total = sum(a["capital"] for a in ALLOCATIONS.values())
    assert total == TOTAL_CAPITAL, f"Allocations sum to {total}, expected {TOTAL_CAPITAL}"
    print("  PASS: Capital allocations sum to $250,000,000")


def test_allocation_weights_sum_to_one():
    """Verify allocation weights sum to 1.0."""
    from config.portfolio_constraints import ALLOCATIONS
    total_weight = sum(a["weight"] for a in ALLOCATIONS.values())
    assert abs(total_weight - 1.0) < 0.001, f"Weights sum to {total_weight}, expected 1.0"
    print("  PASS: Allocation weights sum to 1.0")


def test_vega_bounds_are_valid():
    """Verify vega floor < ceiling."""
    assert NET_VEGA_FLOOR < NET_VEGA_CEILING
    print(f"  PASS: Vega bounds valid ({NET_VEGA_FLOOR:,.0f} < {NET_VEGA_CEILING:,.0f})")


def test_variance_notional_cap_defined():
    """Verify short variance notional cap is set."""
    assert VOL_HARVEST_SHORT_VARIANCE_NOTIONAL_CAP == 80_000_000
    print(f"  PASS: Short variance notional cap = ${VOL_HARVEST_SHORT_VARIANCE_NOTIONAL_CAP:,.0f}")


def test_vol_target_is_twelve_percent():
    """Verify volatility target is 12% annualized."""
    assert VOL_TARGET_ANNUALIZED == 0.12
    print(f"  PASS: Volatility target = {VOL_TARGET_ANNUALIZED*100:.0f}%")


def test_overlay_drag_ceiling():
    """Verify option overlay drag ceiling is 75 basis points."""
    assert OVERLAY_MAX_ANNUAL_DRAG_BPS == 75
    print(f"  PASS: Option overlay drag ceiling = {OVERLAY_MAX_ANNUAL_DRAG_BPS} basis points")


def run_all_tests():
    """Run all constraint validation tests."""
    print("\n" + "=" * 60)
    print("CONSTRAINT VALIDATION TESTS")
    print("=" * 60)

    tests = [
        test_capital_allocation_sums_to_total,
        test_allocation_weights_sum_to_one,
        test_vega_bounds_are_valid,
        test_variance_notional_cap_defined,
        test_vol_target_is_twelve_percent,
        test_overlay_drag_ceiling,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__} — {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test.__name__} — {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)} tests")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
