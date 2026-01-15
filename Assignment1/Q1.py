# Question 1: Message Counter (Immutable Parameter)
# This program demonstrates how immutable types (integers) behave as default parameters.

def count_message(msg, count=0):
    """
    Accepts a message and a count, increments the count, 
    and prints the updated value.
    """
    # Incrementing the count (an immutable integer)
    count += 1

    # Print the message along with the current count
    print(f"Message: {msg} | Updated Count: {count}")

    # Return the updated count so it can be passed back into the next call
    return count

# --- Testing the function ---

# First call: count starts at default (0) and becomes 1
current_count = count_message("heya")

# Second call: we pass the returned value (1) back in, so it becomes 2
current_count = count_message("hello again", current_count)

# Third call: passing the updated value (2) to get 3
current_count = count_message("final message", current_count)
