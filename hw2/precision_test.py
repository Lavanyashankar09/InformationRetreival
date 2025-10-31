from hw2 import precision_at 

recall_ = 0.6  
# fmt: off  
results_ = [2, 1, 3, 4, 7, 5, 6, 8, 9, 10, 11, 12, 13, 14]  # Lucas numbers for debugging.  
# fmt: on  

relevant_ = [1, 14, 2, 3, 5, 8, 13]  # Fibonacci numbers for debugging.

print(precision_at(recall_, results_, relevant_))
# Should be in the ballpark of 0.66.