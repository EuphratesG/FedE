import re

text = """
Value of the 1st triple's likelihood = 0.2100, Object Entity: laboratory or test result
Value of the 2nd triple's likelihood = 0.1500, Object Entity: organism attribute
Value of the 3rd triple's likelihood = 0.3000, Object Entity: cell or molecular dysfunction
Value of the 4th triple's likelihood = 0.4500, Object Entity: therapeutic or preventive procedure
Value of the 5th triple's likelihood = 0.3500, Object Entity: sign or symptom
Value of the 6th triple's likelihood = 0.1800, Object Entity: occupational activity
Value of the 7th triple's likelihood = 0.1200, Object Entity: anatomical abnormality
Value of the 8th triple's likelihood = 0.1000, Object Entity: hormone
"""

likelihoods = re.findall(r'likelihood = (\d+\.\d+)', text)
print(likelihoods)
