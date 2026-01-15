# Question 4: Fix the remove_even function
# This program correctly removes even numbers without skipping elements.

def remove_even(numbers):
    """
    Fix: Instead of removing while iterating (which skips items),
    we use a list comprehension to create a new list of odd numbers.
    """
    # This creates a new list containing only the numbers where n % 2 is not 0
    numbers = [n for n in numbers if n % 2 != 0]

    return numbers

# --- Testing the function ---

# [span_6](start_span)Input as per assignment instructions[span_6](end_span)
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Expected output: [1, 3, 5, 7, 9]
result = remove_even(nums)
print(f"Original list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]")
print(f"Filtered list: {result}")
