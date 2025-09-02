def is_leap(year):
    # Leap year logic
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# Read input
year = int(input())
print(is_leap(year))
