def find_equivalent_basket_combinations(limits):
    # Define the composition of each basket
    basket1 = {"croissants": 6, "jams": 3, "djembes": 1}
    basket2 = {"croissants": 4, "jams": 2, "djembes": 0}
    
    # Initialize a dictionary to track unique combinations by their total products
    unique_combinations = {}
    
    # Function to check if a combination is valid (all products within limits)
    def is_valid_combination(c, j, d, b1, b2):
        return (
            -limits["croissants"] <= c <= limits["croissants"] and
            -limits["jams"] <= j <= limits["jams"] and
            -limits["djembes"] <= d <= limits["djembes"] and
            -limits["basket1"] <= b1 <= limits["basket1"] and
            -limits["basket2"] <= b2 <= limits["basket2"]
        )
    
    # Function to calculate total products in a combination
    def get_product_totals(c, j, d, b1, b2):
        total_c = c + (basket1["croissants"] * b1) + (basket2["croissants"] * b2)
        total_j = j + (basket1["jams"] * b1) + (basket2["jams"] * b2) 
        total_d = d + (basket1["djembes"] * b1) + (basket2["djembes"] * b2)
        return (total_c, total_j, total_d)
    
    # Generate all possible combinations within limits
    for b1 in range(-limits["basket1"], limits["basket1"] + 1):
        for b2 in range(-limits["basket2"], limits["basket2"] + 1):
            for c in range(-limits["croissants"], limits["croissants"] + 1, 10):  # Step by 10 to reduce computation
                for j in range(-limits["jams"], limits["jams"] + 1, 10):  # Step by 10 to reduce computation
                    for d in range(-limits["djembes"], limits["djembes"] + 1):
                        if is_valid_combination(c, j, d, b1, b2):
                            # Calculate the total products
                            totals = get_product_totals(c, j, d, b1, b2)
                            
                            # Add this combination to our dictionary
                            if totals not in unique_combinations:
                                unique_combinations[totals] = []
                            
                            unique_combinations[totals].append((c, j, d, b1, b2))


    return unique_combinations

# Define the limits
limits = {
    "croissants": 250,
    "jams": 350, 
    "djembes": 60,
    "basket1": 60,
    "basket2": 100
}

# Since the full computation would be very large, let's use a reduced set of limits for demonstration
demo_limits = {
    "croissants": 50,  # Reduced from 250
    "jams": 50,        # Reduced from 350
    "djembes": 10,     # Reduced from 60
    "basket1": 10,     # Reduced from 60
    "basket2": 10      # Reduced from 100
}

# Uncomment the line below to run with full limits (warning: this will take a long time and use a lot of memory)
# find_equivalent_basket_combinations(limits)

# Run with reduced limits for demonstration
unique_combinations = find_equivalent_basket_combinations(demo_limits)
print(len(unique_combinations))

# # If you want to focus on specific examples:
# def check_specific_example():
#     # Example: 2 BASKET1 = 3 BASKET2 + 2 DJEMBE
#     basket1 = {"croissants": 6, "jams": 3, "djembes": 1}
#     basket2 = {"croissants": 4, "jams": 2, "djembes": 0}
    
#     # Calculate totals for each combination
#     combo1_totals = (
#         0 + (2 * basket1["croissants"]) + (0 * basket2["croissants"]),
#         0 + (2 * basket1["jams"]) + (0 * basket2["jams"]),
#         0 + (2 * basket1["djembes"]) + (0 * basket2["djembes"])
#     )
    
#     combo2_totals = (
#         0 + (0 * basket1["croissants"]) + (3 * basket2["croissants"]),
#         0 + (0 * basket1["jams"]) + (3 * basket2["jams"]),
#         2 + (0 * basket1["djembes"]) + (0 * basket2["djembes"])
#     )
    
#     print("\nChecking specific example: 2 BASKET1 = 3 BASKET2 + 2 DJEMBE")
#     print(f"Combination 1 (2 BASKET1) totals: {combo1_totals[0]} croissants, {combo1_totals[1]} jams, {combo1_totals[2]} djembes")
#     print(f"Combination 2 (3 BASKET2 + 2 DJEMBE) totals: {combo2_totals[0]} croissants, {combo2_totals[1]} jams, {combo2_totals[2]} djembes")
#     print(f"Are they equivalent? {combo1_totals == combo2_totals}")

# check_specific_example()