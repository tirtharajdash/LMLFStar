from search import Hypothesis

factors = [lambda x: x.get('CNNaffinity')]
experiment = [[2, 10]]  # Single range

h = Hypothesis(factors, experiment)

# Test molecules
molecule_1 = {'CNNaffinity': 6.5}  # Within range
molecule_2 = {'CNNaffinity': 1.5}  # Outside range
molecule_3 = {'MolWt': 320}        # Missing CNNaffinity key
molecule_4 = {'MolWt': 501, 'CNNaffinity': 7.0} #Two keys both within range
molecule_5 = {'MolWt': 320, 'CNNaffinity': 7.0} #Two keys one within range

print(h(molecule_1))  # True
print(h(molecule_2))  # False
print(h(molecule_3))  # False
print(h(molecule_4))  # True
print(h(molecule_5))  # False
