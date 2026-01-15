# Question 2: Order History Tracker (Mutable Parameter)
# This program demonstrates how to safely handle mutable default parameters (lists).

def add_order(order_id, orders=None):
    """
    Accepts an order ID and an optional list of orders.
    Uses 'None' as a default to avoid the shared-list pitfall.
    """
    # Check if orders is None; if so, start a fresh list
    if orders is None:
        orders = []

    # Add the new order ID to the list
    orders.append(order_id)

    # Print the current state of history for clarity
    print(f"Added Order ID: {order_id} | Current History: {orders}")

    # Return the complete order history
    return orders

# --- Testing the function ---

# First call: Starts a new history
history = add_order(101)

# Second call: Pass the previous history back in to update it
history = add_order(102, history)

# Third call: Continue tracking history
history = add_order(103, history)
