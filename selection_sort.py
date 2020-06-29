def selection_sort(array):
	'''
	input is an array of numbers
	'''

	for current in range(0, len(array)-1):
		#create min index
		min_index = current
		#look at every element after current
		for i in range(current+1, len(array)):
			if array[i] < array[min_index]:
				min_index = i

			#swap minindex and current
			temp = array[current]
			array[current] = array[min_index]
			array[min_index] = temp

	return array

print(selection_sort([0,9,1,8,2,7,3,6,5]))

def insertion_sort(array,current):
	'''
	array is a partiall sorted array of numbers
	current is the start of unsorted
	could also combine i>1 and array[i-1] > array[i]
	'''
	i = current
	while i > 1:
		if array[i-1] > array[i]:
			#swap
			temp = array[current]
			array[current] = array[i-1]
			array[i-1] = temp
		else:
			continue
		i -= 1
	return array

##also written another way

def insertion_sort(array):
	for current in range(0,len(array)):
		i = current
		while (i > 0) and (array[i] < array[i-1]):
			#swap
			temp = array[i]
			array[i] = array[i-1]
			array[i-1] = temp
			i -= 1
	return array

print(insertion_sort([2,0,2,1,1,0]))


'''
f(n) = c \cdot k^3
n = k^2
k = sqrt(n)

f(n) = c \cdot n^{3/2}

formular for (a+b)^3:

(a^3 + 3a^2 + 3a + b^2)

'''

