

```python
def basic_operations(a, b):
    return {
        'addition': a + b,
        'subtraction': a - b,
        'multiplication': a * b,
        'division': a / b if b != 0 else 'undefined'
    }
```

```python
def perform_calculations(a, b):
    addition = a + b
    subtraction = a - b
    multiplication = a * b
    division = a / b if b != 0 else None
    return addition, subtraction, multiplication, division
```

```python
def display_results(results):
    for result in results:
        print(result)

# Example usage:
results = ["Result 1", "Result 2", "Result 3"]
display_results(results)
```

```python
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."
    except TypeError:
        return "Error: Invalid input type."
```

```python
def is_even(number):
    """Check if a number is even."""
    return number % 2 == 0

# Example usage:
# print(is_even(4))  # True
# print(is_even(5))  # False
```

```python
def add(a, b):
    """
    Adds two numbers and returns the result.

    Parameters:
    a (int or float): The first number to add.
    b (int or float): The second number to add.

    Returns:
    int or float: The sum of the two numbers.
    """
    return a + b
```

```python
def deploy_application(server, app_name):
    print(f"Deploying {app_name} to {server}...")
    # Simulate deployment process
    success = True  # Assume deployment is successful
    if success:
        print(f"{app_name} deployed successfully on {server}.")
    else:
        print(f"Failed to deploy {app_name} on {server}.")
```

```python
def collect_feedback(feedback_list, new_feedback):
    """
    Collects user feedback and adds it to the existing list.

    Parameters:
    feedback_list (list): The list containing existing feedback.
    new_feedback (str): The new feedback to be added.

    Returns:
    list: Updated list of feedback.
    """
    feedback_list.append(new_feedback)
    return feedback_list

# Example usage:
feedbacks = ["Great app!", "Needs more features."]
updated_feedbacks = collect_feedback(feedbacks, "User-friendly interface.")
print(updated_feedbacks)
```