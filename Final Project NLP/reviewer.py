f = open("original_rt_snippets.txt", "r")
g = open("negative_review.txt", "a")
h = open("positive_review.txt", "a")
j = open("typo.txt", "a")
for x in f:
	print(x)
	y = input()
	if y == 'b':
		g.write(x)
	elif y == 'g':
		h.write(x)
	elif y == 'end':
		break
	else:
		j.write(x)
f.close()
g.close()
h.close()
j.close()