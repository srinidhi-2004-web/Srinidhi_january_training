# Question 3: Fix the Bug (Default Parameter Pitfall)
# This program fixes the issue where a shared list is used across function calls.

def save_error(error, errors=None):
    """
    Rewrite: Uses None as the default value to ensure a new list 
    is created for every unique call unless a list is provided.
    """
    # If no list is provided, create a new local list
    if errors is None:
        errors = []

    errors.append(error)
    return errors

# --- Testing the corrected function ---

# Each call now results in a fresh list, showing the pitfall is fixed
print(f"Call 1: {save_error('E1')}") # Expected: ['E1']
print(f"Call 2: {save_error('E2')}") # Expected: ['E2']
print(f"Call 3: {save_error('E3')}") # Expected: ['E3']

# If we WANT to track history, we pass the list back in
history = save_error("E4")
history = save_error("E5", history)
print(f"Intentional History: {history}") # Expected: ['E4', 'E5']
