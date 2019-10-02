from make_csvs import *

results = [ i.replace("\n","") for i in open("results").readlines()]
profiles = [ i.replace("\n","") for i in open("profiles").readlines()]

metric1 = {}
metric2 = {} 
metric3 = {}

for result in results:
	key = ",".join(result.split(",")[:-1])
	print(result)
	try:
		metric1[key] = parse_result(result)
	except:
		print("Failure")
		print(key)
for profile in profiles:
	key = ",".join(profile.split(",")[:-1])
	print(profile)
	metric2[key] = parse_profile(profile)

if set(metric1.keys()) != set(metric2.keys()):
	print("Profiles is missing:")
	print(set(metric1.keys())-set(metric2.keys()))
	print(set(metric2.keys())-set(metric1.keys()))

delete_keys = set()
for key in metric1:
	if key[-2:] == "20":
		delete_keys.add(key)
for key in metric2:
	if key[-2:] == "20":
		delete_keys.add(key)

#for key in delete_keys:
#	metric1.pop(key,None)
#	metric2.pop(key,None)

print(max(metric1, key=metric1.get),metric1[max(metric1, key=metric1.get)])
print(max(metric2, key=metric2.get),metric2[max(metric2, key=metric2.get)])
