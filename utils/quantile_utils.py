# In a new or existing file: utils/quantile_utils.py
import numpy as np

def get_standard_quantile_levels(num_quantiles: int):
    """
    Generates a standard set of quantile levels, typically symmetric around 0.5.
    Ensures an odd number of quantiles to include the median (0.5).
    """
    if num_quantiles % 2 == 0:
        num_quantiles += 1 # Ensure odd number to include median
        print(f"Adjusted num_quantiles to {num_quantiles} to include median.")

    if num_quantiles == 1:
        return [0.5]

    # Generate symmetric quantiles
    # Example: for num_quantiles = 3 -> [0.1, 0.5, 0.9] (approx)
    # Example: for num_quantiles = 5 -> [0.1, 0.25, 0.5, 0.75, 0.9] (approx)

    # A simple way to generate somewhat standard quantiles:
    if num_quantiles == 3:
        return [0.1, 0.5, 0.9]
    elif num_quantiles == 5:
        return [0.1, 0.25, 0.5, 0.75, 0.9]
    elif num_quantiles == 7:
        return [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
    elif num_quantiles == 9:
        return [0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95]
    else: # Fallback for other odd numbers
        # This creates evenly spaced points including 0.5
        # and then tries to make them somewhat symmetric.
        # More sophisticated logic might be needed for "standard" financial quantiles.
        half_points = (num_quantiles - 1) // 2
        lower_quantiles = np.linspace(0.05, 0.45, half_points, endpoint=True).tolist()
        upper_quantiles = np.linspace(0.55, 0.95, half_points, endpoint=True).tolist()
        return sorted(list(set(lower_quantiles + [0.5] + upper_quantiles)))

def generate_quantile_levels(num_quantiles: int, coverage_range: tuple = (0.05, 0.95)):
    """
    Generates quantile levels based on the number of quantiles and a coverage range.
    Ensures the median (0.5) is included if num_quantiles is odd.
    """
    if num_quantiles < 1:
        raise ValueError("Number of quantiles must be at least 1.")

    min_q, max_q = coverage_range
    if not (0 < min_q < 0.5 and 0.5 < max_q < 1):
        raise ValueError("Coverage range must be (min_q, max_q) with 0 < min_q < 0.5 and 0.5 < max_q < 1.")

    if num_quantiles == 1:
        return [0.5]

    # Ensure median is included if num_quantiles is odd
    include_median = (num_quantiles % 2 != 0)

    points_per_side = (num_quantiles - (1 if include_median else 0)) // 2

    lower_quantiles = np.linspace(min_q, 0.5, points_per_side + 1, endpoint=False)[0:points_per_side].tolist() if points_per_side > 0 else []
    upper_quantiles = np.linspace(0.5, max_q, points_per_side + 1, endpoint=True)[1:].tolist() if points_per_side > 0 else []

    levels = sorted(list(set(lower_quantiles + ([0.5] if include_median else []) + upper_quantiles)))

    # If due to linspace precision, we don't get exactly num_quantiles, adjust.
    # This is a simple adjustment; more robust might be needed.
    if len(levels) != num_quantiles:
        # Fallback to a simpler linspace if the above logic fails for some num_quantiles
        levels = np.linspace(min_q, max_q, num_quantiles).tolist()
        if include_median and 0.5 not in levels:
            # Force include median and re-sort, then pick num_quantiles
            levels.append(0.5)
            levels = sorted(list(set(levels)))
            if len(levels) > num_quantiles: # If too many, try to pick symmetrically
                # This part can be tricky, for now, let's just take the outer and inner ones
                # A more robust method would be to ensure 0.5 is there and then pick closest points.
                # For simplicity, if this fallback is hit, it might not be perfectly symmetrical.
                excess = len(levels) - num_quantiles
                levels = levels[excess//2 : len(levels) - (excess - excess//2)]


    return [round(q, 4) for q in levels] # Round for cleaner output

def quantile_levels_to_string(quantile_levels: list):
    """Converts a list of quantile levels to a string."""
    if not quantile_levels:
        return ""
    return ",".join(map(str, quantile_levels))

def describe_quantiles(quantile_levels: list):
    """Provides a human-readable description of the quantile levels."""
    if not quantile_levels:
        return "No quantiles selected."

    n = len(quantile_levels)
    description = f"{n} quantiles: {quantile_levels}. "

    if 0.5 in quantile_levels:
        description += "Includes median (P50). "

    if n >= 3:
        lower_outer = min(quantile_levels)
        upper_outer = max(quantile_levels)
        coverage = (upper_outer - lower_outer) * 100
        description += f"Covers approx. {coverage:.0f}% prediction interval ({lower_outer*100:.0f}th to {upper_outer*100:.0f}th percentile)."

    return description.strip()

